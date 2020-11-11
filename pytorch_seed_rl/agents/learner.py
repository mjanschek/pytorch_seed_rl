# Copyright 2020 Michael Janschek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent that runs inference and learning in parallel via multiple threads.

Consists of:
    # . n inference threads
    # . 1 training thread (python main thread)
    # . l data prefetching threads
    # . 1 reporting/logging object
"""
import copy
import gc
import pprint
import sys
import time
from collections import deque
from queue import Empty

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.distributions import Categorical
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel

from .. import agents
from ..data_structures.trajectory_store import TrajectoryStore
from ..functional import util, vtrace


def _gen_key(actor_name, env_id):
    return actor_name+"_env{}".format(env_id)


class Learner():
    """Agent that runs inference and learning in parallel via multiple threads.

    # . Runs inference for observations received from :py:class:`~pytorch_seed_rl.agents.Actor`s.
    # . Puts incomplete trajectories to :py:class:`~pytorch_seed_rl.data_structures.trajectory_store`
    # . Trains global model from trajectories received from a data prefetching thread.
    """

    def __init__(self,
                 rank,
                 num_learners,
                 num_actors,
                 env_spawner,
                 model,
                 optimizer,
                 inference_batchsize=4,
                 training_batchsize=4,
                 rollout_length=80,
                 max_epoch=-1,
                 max_steps=-1,
                 max_time=-1,
                 save_path=".",
                 exp_name=""):

        # save arguments as attributes where needed
        self.rollout_length = rollout_length

        self.inference_batchsize = inference_batchsize
        self.training_batchsize = training_batchsize

        self.total_num_envs = num_actors*env_spawner.num_envs

        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.max_time = max_time

        self.save_path = save_path
        self.exp_name = exp_name

        assert inference_batchsize <= self.total_num_envs

        # set counters
        self.inference_epoch = 0
        self.inference_steps = 0
        self.inference_time = 0

        self.training_epoch = 0
        self.training_steps = 0
        self.training_time = 0

        self.fetching_time = 0
        self.actual_fetching_time = 0

        self.runtime = 0
        self.mean_latency = 0
        self.episodes_seen = 0

        # torch
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        try:
            self.inference_model = model = copy.deepcopy(self.model)
            self.inference_model.eval()
        except Exception as e:
            print(e)
        if torch.cuda.device_count() > 1:
            self.model = DistributedDataParallel(self.model)
            self.model.share_memory()
        self.optimizer = optimizer

        def linear_lambda(epoch):
            return 1 - min(epoch * rollout_length * training_batchsize, max_steps) / max_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, linear_lambda)

        # rpc stuff
        self.active_envs = {}
        self.shutdown = False
        #self.rref = RRef(self)

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.lock = mp.Lock()
        self.start_training_event = mp.Event()
        self.training_lock = mp.Lock()
        self.model_lock = mp.Lock()
        self.pending_rpcs = deque(maxlen=self.total_num_envs)
        self.future_answers = [Future() for i in range(self.total_num_envs)]

        self.env_info = env_spawner.env_info

        # init attributes
        self.dummy_obs = env_spawner.dummy_obs
        self.dummy_obs['action'] = torch.zeros(1, 1)
        self.dummy_obs['baseline'] = torch.zeros(1, 1)
        self.dummy_obs['policy_logits'] = torch.zeros(
            1, 1, self.env_info['action_space'].n)

        dummy_frame = self.dummy_obs['frame']
        self.inference_memory = torch.zeros_like(dummy_frame, device=self.device) \
            .repeat((1, self.inference_batchsize) + (1,)*(len(dummy_frame.size())-2))

        self.training_batch = {}

        # spawn trajectory store
        self.trajectory_store = TrajectoryStore(self.total_num_envs,
                                                self.dummy_obs,
                                                self.device,
                                                max_trajectory_length=rollout_length)

        # spawn actors
        agent_rref = rpc.RRef(self)
        self._spawn_actors(agent_rref,
                           num_learners,
                           num_actors,
                           env_spawner)

        self.logger = agents.Logger(
            ['csv'], ['episodes', 'training', 'system'], "/".join([self.save_path, "logs", self.exp_name]))

        # start prefetch thread
        self.batch_queue = mp.Queue(maxsize=1)
        self.prefetch_thread = agent_rref.remote().prefetch()

    @staticmethod
    def _spawn_actors(agent_rref, num_learners, num_actors, env_spawner):
        for i in range(num_actors):
            actor_info = rpc.get_worker_info("actor%d" % (i+num_learners))
            actor_rref = rpc.remote(actor_info,
                                    agents.Actor,
                                    args=(i,
                                          agent_rref,
                                          env_spawner))
            actor_rref.remote().loop()
        print("%d actors spawned, %d environments each." %
              (num_actors, env_spawner.num_envs))

    @async_execution
    def batched_inference(self, caller_id, state, metrics=None):
        f_answer = self.future_answers[caller_id].then(
            lambda f_answers: f_answers.wait())

        with self.lock:
            self.pending_rpcs.append((caller_id, state, metrics))
            if len(self.pending_rpcs) == self.inference_batchsize:
                process_rpcs = [self.pending_rpcs.popleft()
                                for _ in range(self.inference_batchsize)]

                start = time.time()
                self.process_batch(process_rpcs)
                self.inference_time += time.time() - start

        return f_answer

    def process_batch(self, pending_rpcs):

        states = listdict_to_dictlist([s for _, s, _ in pending_rpcs])
        for k, v in states.items():
            states[k] = torch.cat(v, dim=1).to(self.device)

        with self.model_lock:
            inference_output, _ = self.inference_model(states)

        states = {k: v.cpu().detach()
                  for k, v in {**states, **inference_output}.items()}

        self.inference_steps += self.inference_batchsize
        self.inference_epoch += 1

        # pylint: disable=not-callable
        timestamp = torch.tensor(time.time(), dtype=torch.float64).view(1, 1) 
        latencies = []
        for i, rpc_tuple in enumerate(pending_rpcs):
            caller_id, _, metrics = rpc_tuple
            state = {k: v[0, i].view(self.dummy_obs[k].shape)
                     for k, v in states.items()}

            latencies.append(metrics['latency'])
            metrics['timestamp'] = timestamp.clone()
            self.trajectory_store.add_to_entry(caller_id, state, metrics)

            action = inference_output['action'][0][i].view(1, 1).cpu().detach()

            f_answers = self.future_answers[caller_id]
            self.future_answers[caller_id] = Future()
            f_answers.set_result((action,
                                  self.shutdown,
                                  caller_id,
                                  dict()
                                  ))

    def _run_inference(self):
        n_actions = self.env_info['action_space'].n
        prob = 1/n_actions

        policies = torch.tensor(prob).repeat(  # pylint: disable=not-callable
            self.inference_batchsize, n_actions).to(self.device)
        values = torch.rand(self.inference_batchsize).to(self.device)
        m = Categorical(policies)
        actions = m.sample().to(self.device)

        return actions, policies, values

    def loop_training(self):
        self.t_start = time.time()

        print("Waiting for training start...")

        while not (self.shutdown or (self.training_epoch > self.max_epoch > 0) or
                   (self.training_steps > self.max_steps > 0) or
                   (self.training_time > self.max_time > 0)):

            self.start_training_event.wait()
            with self.training_lock:
                self.batched_training()

                with self.model_lock:
                    with torch.no_grad():
                        for m, i in zip(self.model.parameters(), self.inference_model.parameters()):
                            i.copy_(m)

                self.start_training_event.clear()

            system_metrics = {
                "runtime": self._get_runtime(),
                "fetching_time": self.fetching_time,
                "inference_time": self.inference_time,
                "inference_steps": self.inference_steps,
                "training_time": self.training_time,
                "training_steps": self.training_steps,
                "drop_off_queuesize": len(self.trajectory_store.drop_off_queue),
                "batch_queuesize": self.batch_queue.qsize(),
                "pending_rpcs": len(self.pending_rpcs)
            }
            self.logger.log('system', system_metrics)

        self.shutdown = True
        self._cleanup()
        self.report()

    def _get_runtime(self):
        return time.time()-self.t_start

    def batched_training(self):
        """Trains on sampled, prefetched trajectories.
        """
        start = time.time()
        training_metrics = self._learn_from_batch(self.training_batch)

        self.training_time += time.time()-start

        self.logger.log('training', training_metrics)

        if self.training_epoch % 100 == 0:
            print(pprint.pformat(training_metrics))

    def _log_trajectory(self, trajectory):
        gti = trajectory['global_trajectory_id']
        gei = trajectory['global_episode_id']

        self.episodes_seen += 1

        if trajectory['complete'].item() is True:
            i = trajectory['current_length']-1
            states = trajectory['states']
            metrics = trajectory['metrics']
            self.mean_latency = self.mean_latency + \
                (metrics['latency'].mean().item() -
                 self.mean_latency) / self.episodes_seen

            episode_data = {
                'global_episode_id': gei,
                'training_steps': self.training_steps,
                'return': states['episode_return'][i],
                'length': states['episode_step'][i],
                'timestamp': metrics['timestamp'][i-1],
                'mean_latency': metrics['latency'].mean()
            }
            self.logger.log('episodes', episode_data)

    def _to_batch(self, trajectories):
        states = listdict_to_dictlist([t['states'] for t in trajectories])

        for k, v in states.items():
            states[k] = torch.cat(v, dim=1)

        states['current_length'] = torch.stack(
            [t['current_length'] for t in trajectories])

        return states

    def _learn_from_batch(self,
                          batch,
                          baseline_cost=0.5,
                          entropy_cost=0.01,
                          discounting=0.99,
                          reward_clipping="abs_one",
                          grad_norm_clipping=40.):
        learner_outputs, _ = self.model(batch)

        batch_length = batch['current_length'].sum().item()

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1]
                           for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = util.compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        # baseline_loss = baseline_cost * util.compute_baseline_loss(
        #     vtrace_returns.vs - learner_outputs["baseline"]
        # )

        baseline_loss = F.mse_loss(
            learner_outputs["baseline"], vtrace_returns.vs, reduction='sum')

        entropy_loss = util.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss \
            + baseline_cost * baseline_loss \
            + entropy_cost * entropy_loss

        self.training_steps += batch_length
        self.training_epoch += 1

        # pylint: disable=not-callable
        timestamp = torch.tensor(time.time(), dtype=torch.float64).view(1, 1)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm_clipping)
        self.optimizer.step()
        self.scheduler.step()

        stats = {
            "runtime": self._get_runtime(),
            "training_time": self.training_time,
            "training_epoch": self.training_epoch,
            "training_steps": self.training_steps,
            "total_loss": total_loss.detach().cpu().item(),
            "pg_loss": pg_loss.detach().cpu().item(),
            "baseline_loss": baseline_loss.detach().cpu().item(),
            "entropy_loss": entropy_loss.detach().cpu().item(),
        }

        return stats

    def _answer_rpcs(self):
        while len(self.active_envs) > 0:
            try:
                rpc_tuple = self.pending_rpcs.popleft()
            except IndexError:
                time.sleep(0.1)
                continue
            caller_id, _, _ = rpc_tuple
            # print(i, caller_id)
            self.future_answers[caller_id].set_result(
                (self.dummy_obs['last_action'],
                 True,
                 caller_id,
                 dict()))

    def _cleanup(self):
        # Answer pending rpcs to enable actors to terminate
        self._answer_rpcs()
        print("Pendings rpcs answered")

        # empty self.batch_queue
        while not self.batch_queue.empty():
            self.batch_queue.get()

        self.logger.write_buffers()

        # Remove process to ensure freeing of resources.
        self.prefetch_thread.to_here()
        del self.prefetch_thread
        print("Prefetch rpc joined")

        # Shutdown and remove queue
        self.batch_queue.close()
        self.batch_queue.join_thread()
        del self.batch_queue
        print("self.batch_queue joined")

        # Run garbage collection to ensure freeing of resources.
        gc.collect()

    def report(self):
        """Reports data to CLI
        """
        time = self._get_runtime()
        if time > 0:
            print("\n============== REPORT ==============")
            fps = self.inference_steps / time

            print("infered", str(self.inference_steps), "times")
            print("in", str(time), "seconds")
            print("==>", str(fps), "fps")

            fps = self.training_steps / time
            print("trained", str(self.training_steps), "times")
            print("in", str(time), "seconds")
            print("==>", str(fps), "fps")

            print("Total inference_time:", str(
                self.inference_time), "seconds")

            print("Total training_time:", str(
                self.training_time), "seconds")

            print("Total fetching_time:", str(
                self.fetching_time), "seconds")

            print("Mean inference latency:", str(
                self.mean_latency), "seconds")

    def prefetch(self):
        """prefetches data from inference thread
        """
        while not self.shutdown:
            start = time.time()
            # with self.training_lock:
            if len(self.trajectory_store.drop_off_queue) >= self.training_batchsize:
                actual_start = time.time()
                trajectories = []
                for _ in range(self.training_batchsize):
                    t = self.trajectory_store.drop_off_queue.popleft()
                    self._log_trajectory(t)
                    trajectories.append(t)

                batch = self._to_batch(trajectories)

                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                with self.training_lock:
                    self.training_batch = batch

                self.start_training_event.set()
                self.fetching_time += time.time() - start
                self.actual_fetching_time += time.time() - actual_start

            time.sleep(0.1)

    def check_in(self, caller_id):
        self.active_envs[caller_id] = True

    def check_out(self, caller_id):
        del self.active_envs[caller_id]


def listdict_to_dictlist(listdict):
    return {k: [dic[k] for dic in listdict] for k in listdict[0]}


def dictlist_to_listdict(dictlist):
    return [dict(zip(dictlist, t)) for t in zip(*dictlist.values())]
