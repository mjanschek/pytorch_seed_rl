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
from queue import Empty
import gc
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.distributions import Categorical
from torch.futures import Future

from ..data_structures.trajectory_store import TrajectoryStore
from .. import agents


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
                 loss,
                 inference_batchsize=16,
                 training_batchsize=8,
                 rollout_length=64,
                 max_epoch=-1,
                 max_steps=1000000,
                 max_time=10,
                 save_path="."):

        # save arguments as attributes where needed
        self.rollout_length = rollout_length

        self.inference_batchsize = inference_batchsize
        self.training_batchsize = training_batchsize

        self.total_num_envs = num_actors*env_spawner.num_envs

        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.max_time = max_time

        self.save_path = save_path

        assert inference_batchsize <= self.total_num_envs

        # set counters
        self.inference_steps = 0
        self.training_epoch = 0
        self.training_steps = 0
        self.training_time = 0
        self.final_time = 0

        # torch
        self.device = torch.device('cuda')

        # rpc stuff
        self.active_envs = {}
        self.shutdown = False
        #self.rref = RRef(self)

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.lock = mp.Lock()
        self.pending_rpcs = deque(maxlen=self.total_num_envs)
        self.future_answers = [Future() for i in range(self.total_num_envs)]

        self.env_info = env_spawner.env_info

        # init attributes
        self.dummy_obs = env_spawner.dummy_obs
        self.dummy_obs['last_policy'] = torch.zeros(
            1, self.env_info['action_space'].n)
        self.dummy_obs['last_value'] = torch.zeros(1, 1)

        dummy_frame = self.dummy_obs['frame']
        self.inference_memory = torch.zeros_like(dummy_frame) \
            .repeat((1, self.inference_batchsize) + (1,)*(len(dummy_frame.size())-2)) \
            .cuda()

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
            ['csv'], ['episodes'], "/".join([self.save_path, "logs"]))

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

    @staticmethod
    @async_execution
    def batched_inference(rref, caller_id, state, metrics=None):
        self = rref.local_value()

        f_answer = self.future_answers[caller_id].then(
            lambda f_answers: f_answers.wait())

        with self.lock:
            self.pending_rpcs.append((caller_id, state, metrics))
            if len(self.pending_rpcs) == self.inference_batchsize:
                process_rpcs = [self.pending_rpcs.popleft()
                                for _ in range(self.inference_batchsize)]
                self.process_batch(process_rpcs)

        return f_answer

    def process_batch(self, pending_rpcs):

        for i, rpc_tuple in enumerate(pending_rpcs):
            _, state, _ = rpc_tuple
            frame = state['frame']
            self.inference_memory[0, i].copy_(frame[0, 0])

        # placeholder inference

        # pylint: disable=not-callable
        actions, policies, values = self._run_inference()
        ##########

        self.inference_steps += self.inference_batchsize

        timestamp = torch.tensor(time.time(), dtype=torch.float64).view(1, 1)

        for i, rpc_tuple in enumerate(pending_rpcs):
            caller_id, state, metrics = rpc_tuple

            metrics['timestamp'] = timestamp.clone()
            self.trajectory_store.add_to_entry(caller_id, state, metrics)

            action = actions[i].view(1, 1).cpu()
            policy = policies[i].view(1, self.env_info['action_space'].n).cpu()
            value = values[i].view(1, 1).cpu()

            f_answers = self.future_answers[caller_id]
            self.future_answers[caller_id] = Future()
            f_answers.set_result((action,
                                  self.shutdown,
                                  caller_id,
                                  dict(last_policy=policy,
                                       last_value=value)
                                  ))

        self.inference_memory.fill_(0.)

    def _run_inference(self):
        n_actions = self.env_info['action_space'].n
        prob = 1/n_actions

        # pylint: disable=not-callable
        policies = torch.tensor(prob).repeat(
            self.inference_batchsize, n_actions).to(self.device)
        values = torch.rand(self.inference_batchsize).to(self.device)
        m = Categorical(policies)
        actions = m.sample().to(self.device)

        return actions, policies, values

    def loop_training(self):
        start = time.time()

        while not (self.shutdown or (self.training_epoch > self.max_epoch > 0) or
                   (self.training_steps > self.max_steps > 0) or
                   (self.training_time > self.max_time > 0)):
            self.batched_training()

        self.shutdown = True
        self.final_time = time.time()-start
        self._cleanup()
        self.report()

    def batched_training(self):
        """Trains on sampled, prefetched trajecotries.
        """
        start = time.time()

        try:
            batch = self.batch_queue.get(timeout=30)
        except Empty:
            self.training_time += time.time()-start
            return

        # placeholder-training

        # print(trajectories[0])
        # batch = self._to_batch(trajectories)
        training_metrics = self._learn_from_batch(batch)
        ##########
        # self.shutdown = True

        # logger.log_episode(trajectory)

        self.training_time += time.time()-start

    def _log_trajectory(self, trajectory):
        gti = trajectory['global_trajectory_id']
        gei = trajectory['global_episode_id']
        
        

        if trajectory['complete'].item() is True:
            i = trajectory['current_length']-1
            states = trajectory['states']
            metrics = trajectory['metrics']
            episode_data = {
                'global_episode_id': gei,
                'return': states['episode_return'][i],
                'length': states['episode_step'][i],
                'timestamp': metrics['timestamp'][i-1]
            }
            self.logger.log('episodes', episode_data)

    def _to_batch(self, trajectories):
        states = listdict_to_dictlist([t['states'] for t in trajectories])

        for k, v in states.items():
            states[k] = torch.cat(v, dim=1)

        states['current_length'] = torch.stack([t['current_length'] for t in trajectories])

        return states

    def _learn_from_batch(self, batch):
        #batch_shape = batch['reward'].size()[:2]
        self.training_steps += batch['current_length'].sum().item()

        return {}

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
                 dict(last_policy=self.dummy_obs['last_policy'],
                      last_value=self.dummy_obs['last_value'])))

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
        """Reports data to a logging system
        """
        if self.training_time > 0:
            fps = self.inference_steps / self.training_time

            print("infered", str(self.inference_steps), "times")
            print("in", str(self.training_time), "seconds")
            print("==>", str(fps), "fps")

            fps = self.training_steps / self.training_time
            print("trained", str(self.training_steps), "times")
            print("in", str(self.training_time), "seconds")
            print("==>", str(fps), "fps")

    def prefetch(self):
        """prefetches data from inference thread
        """
        while not self.shutdown:
            if len(self.trajectory_store.drop_off_queue) >= self.training_batchsize:
                trajectories = []
                for _ in range(self.training_batchsize):
                    t = self.trajectory_store.drop_off_queue.popleft()
                    self._log_trajectory(t)
                    trajectories.append(t)

                batch = self._to_batch(trajectories)
                self.batch_queue.put(batch)
            time.sleep(0.1)

    def check_in(self, caller_id):
        self.active_envs[caller_id] = True

    def check_out(self, caller_id):
        del self.active_envs[caller_id]


def listdict_to_dictlist(listdict):
    return {k: [dic[k] for dic in listdict] for k in listdict[0]}
