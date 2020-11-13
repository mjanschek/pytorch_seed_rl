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
    #. n inference threads
    #. 1 training thread (python main thread)
    #. l data prefetching threads
    #. 1 reporting/logging object
"""
import copy
import gc
import pprint
# import threading
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from .. import agents
from ..agents.rpc_callee import RpcCallee
from ..data_structures.trajectory_store import TrajectoryStore
from ..functional import util, vtrace
from ..functional.util import listdict_to_dictlist

# def _gen_key(actor_name, env_id):
#     return actor_name+"_env{}".format(env_id)


class Learner(RpcCallee):
    """Agent that runs inference and learning in parallel via multiple threads.

    #. Runs inference for observations received from :py:class:`~pytorch_seed_rl.agents.Actor`s.
    #. Puts incomplete trajectories to :py:class:`~pytorch_seed_rl.data_structures.trajectory_store`
    #. Trains global model from trajectories received from a data prefetching thread.
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

        self.total_num_envs = num_actors*env_spawner.num_envs

        super().__init__(rank,
                         num_callees=num_learners,
                         num_callers=num_actors,
                         caller_class=agents.Actor,
                         caller_args=[env_spawner],
                         rpc_batchsize=inference_batchsize,
                         max_pending_rpcs=self.total_num_envs)

        # save arguments as attributes where needed
        self.rollout_length = rollout_length

        self.rpc_batchsize = inference_batchsize
        self.training_batchsize = training_batchsize

        self.training_batch = {}
        self.states_to_store = {}

        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.max_time = max_time

        self.save_path = save_path
        self.exp_name = exp_name

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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        self.eval_model = copy.deepcopy(self.model)
        self.eval_model.eval()

        if torch.cuda.device_count() > 1:
            self.model = DistributedDataParallel(self.model)
            self.model.share_memory()

        self.optimizer = optimizer

        def linear_lambda(epoch):
            return 1 - min(epoch * rollout_length * training_batchsize, max_steps) / max_steps
        self.scheduler = LambdaLR(self.optimizer, linear_lambda)

        # rpc stuff
        self.event_start_training = mp.Event()
        self.event_start_processing = mp.Event()
        self.training_lock = mp.Lock()
        self.model_lock = mp.Lock()

        self.placeholder_obs = self._build_placeholder_obs(env_spawner)

        # spawn trajectory store
        self.trajectory_store = TrajectoryStore(self.total_num_envs,
                                                self.placeholder_obs,
                                                self.device,
                                                max_trajectory_length=rollout_length)

        self.logger = agents.Logger(
            ['csv'], ['episodes', 'training', 'system'], "/".join([self.save_path, "logs", self.exp_name]))

        # start prefetch thread
        self.batch_queue = mp.Queue(maxsize=1)
        self.prefetch_thread = self.rref.remote().prefetch()

        self._start_callers()

    def _loop(self):
        self.event_start_training.wait()
        with self.training_lock:
            self.batched_training()

            with self.model_lock, torch.no_grad():
                # copy parameters from training to inference model
                for m, i in zip(self.model.parameters(), self.eval_model.parameters()):
                    i.copy_(m)

            self.event_start_training.clear()

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

        self.shutdown = ((self.training_epoch > self.max_epoch > 0) or
                         (self.training_steps > self.max_steps > 0) or
                         (self.training_time > self.max_time > 0))

    @staticmethod
    def _build_placeholder_obs(env_spawner):
        placeholder_obs = env_spawner.placeholder_obs
        placeholder_obs['action'] = torch.zeros(1, 1)
        placeholder_obs['baseline'] = torch.zeros(1, 1)
        placeholder_obs['policy_logits'] = torch.zeros(
            1, 1, env_spawner.env_info['action_space'].n)

        return placeholder_obs

    def process_batch(self, caller_ids, *batch, **misc):
        states = batch[0]
        for k, v in states.items():
            states[k] = torch.cat(v, dim=1).to(self.device)

        start = time.time()

        inference_output, _ = self.eval_model(states)
        self.inference_time += time.time() - start

        self.states_to_store = {k: v.detach()
                                for k, v in {**states, **inference_output}.items()}

        self.inference_steps += self.rpc_batchsize
        self.inference_epoch += 1

        results = {c: inference_output['action'][0][i].view(
            1, 1).cpu().detach() for i, c in enumerate(caller_ids)}

        # add states to store in parallel process
        self.rref.remote().add_to_store(caller_ids, misc['metrics'])

        return results

    def add_to_store(self, caller_ids, all_metrics):
        states = self.states_to_store

        # pylint: disable=not-callable
        timestamp = torch.tensor(time.time(), dtype=torch.float64).view(1, 1)

        latencies = []
        for i, zipped_caller_id_metrics in enumerate(list(zip(caller_ids, all_metrics))):
            caller_id, metrics = zipped_caller_id_metrics

            state = {k: v[0, i].view(self.placeholder_obs[k].shape)
                     for k, v in states.items()}

            latencies.append(metrics['latency'])
            metrics['timestamp'] = timestamp.clone()
            self.trajectory_store.add_to_entry(caller_id, state, metrics)

    def _get_runtime(self):
        return time.time()-self.t_start

    def batched_training(self):
        """Trains on sampled, prefetched trajectories.
        """
        start = time.time()
        training_metrics = self._learn_from_batch(self.training_batch)

        self.training_time += time.time()-start

        self.logger.log('training', training_metrics)

        if self.training_epoch % 10 == 0:
            print(pprint.pformat(training_metrics))

    def _to_batch(self, trajectories):
        states = listdict_to_dictlist([t['states'] for t in trajectories])

        for k, v in states.items():
            states[k] = torch.cat(v, dim=1)

        states['current_length'] = torch.stack(
            [t['current_length'] for t in trajectories])

        return states

    @staticmethod
    def compute_losses(batch,
                       learner_outputs,
                       discounting=0.99,
                       reward_clipping="abs_one"):
        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1]
                           for key, tensor in learner_outputs.items()}

        # clip rewards, if wanted
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

        baseline_loss = F.mse_loss(
            learner_outputs["baseline"],
            vtrace_returns.vs, reduction='sum')

        entropy_loss = util.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        return pg_loss, baseline_loss, entropy_loss

    def _learn_from_batch(self,
                          batch,
                          baseline_cost=0.5,
                          entropy_cost=0.01,
                          grad_norm_clipping=40.):

        batch_length = batch['current_length'].sum().item()
        learner_outputs, _ = self.model(batch)

        pg_loss, baseline_loss, entropy_loss = self.compute_losses(
            batch, learner_outputs)

        total_loss = pg_loss \
            + baseline_cost * baseline_loss \
            + entropy_cost * entropy_loss

        self.training_steps += batch_length
        self.training_epoch += 1

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

    def _cleanup(self):
        super()._cleanup()

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

        self.report()

    def prefetch(self):
        """prefetches data from inference thread
        """
        while not self.shutdown:
            start = time.time()

            if len(self.trajectory_store.drop_off_queue) >= self.training_batchsize:
                actual_start = time.time()
                trajectories = []
                for _ in range(self.training_batchsize):
                    t = self.trajectory_store.drop_off_queue.popleft()
                    self._log_trajectory(t)
                    trajectories.append(t)

                batch = self._to_batch(trajectories)

                # lock training while replacing training batch
                with self.training_lock:
                    self.training_batch = batch
                    self.event_start_training.set()  # start next training

                # update stats
                self.fetching_time += time.time() - start
                self.actual_fetching_time += time.time() - actual_start

            time.sleep(0.1)

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

    def report(self):
        """Reports data to CLI
        """
        runttime = self._get_runtime()
        if runttime > 0:
            print("\n============== REPORT ==============")
            fps = self.inference_steps / runttime

            print("infered", str(self.inference_steps), "times")
            print("in", str(runttime), "seconds")
            print("==>", str(fps), "fps")

            fps = self.training_steps / runttime
            print("trained", str(self.training_steps), "times")
            print("in", str(runttime), "seconds")
            print("==>", str(fps), "fps")

            print("Total inference_time:", str(
                self.inference_time), "seconds")

            print("Total training_time:", str(
                self.training_time), "seconds")

            print("Total fetching_time:", str(
                self.fetching_time), "seconds")

            print("Mean inference latency:", str(
                self.mean_latency), "seconds")
