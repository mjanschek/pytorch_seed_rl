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
import sys
import time
from collections import deque

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
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
                 inference_batchsize=8,
                 training_batchsize=8,
                 rollout_length=64,
                 max_epoch=-1,
                 max_steps=1000000,
                 max_time=60,
                 save_path=""):

        # save arguments as attributes where needed
        self.rollout_length = rollout_length

        self.inference_batchsize = inference_batchsize
        self.training_batchsize = training_batchsize

        self.total_num_envs = num_actors*env_spawner.num_envs

        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.max_time = max_time

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
        self.shutdown = False
        self.rref = RRef(self)

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.lock = mp.Lock()
        self.pending_rpcs = deque(maxlen=self.total_num_envs)
        self.future_answers = [Future() for i in range(self.total_num_envs)]

        # init attributes
        self.dummy_obs = env_spawner.dummy_obs

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
        self.actor_rrefs = self._spawn_actors(agent_rref,
                                              num_learners,
                                              num_actors,
                                              env_spawner)

        # start prefetch thread
        self.prefetch_rpc = self.rref.rpc_async().prefetch()

    @staticmethod
    def _spawn_actors(agent_rref, num_learners, num_actors, env_spawner):
        actor_rrefs = []

        for i in range(num_actors):
            actor_info = rpc.get_worker_info("actor%d" % (i+num_learners))
            actor_rref = rpc.remote(actor_info,
                                    agents.Actor,
                                    args=(i,
                                          agent_rref,
                                          env_spawner))

            actor_rrefs.append(actor_rref)
            print("actor%d spawned" % (i+num_learners))
        return actor_rrefs

    @staticmethod
    @async_execution
    def batched_inference(rref, caller_id, state, metrics=None):
        self = rref.local_value()

        f_answer = self.future_answers[caller_id].then(
            lambda f_answers: f_answers.wait())

        with self.lock:
            self.pending_rpcs.append((caller_id, state))
            if len(self.pending_rpcs) == self.inference_batchsize:
                process_rpcs = [self.pending_rpcs.popleft()
                                for _ in range(self.inference_batchsize)]
                self.process_batch(process_rpcs)

            if self.shutdown:
                for _ in range(len(self.pending_rpcs)):
                    caller_id, _ = self.pending_rpcs.pop()
                    self.future_answers[caller_id].set_result(
                        (self.dummy_obs['last_action'], True, caller_id))

        return f_answer

    def process_batch(self, pending_rpcs):

        for i, rpc_tuple in enumerate(pending_rpcs):
            _, state = rpc_tuple
            frame = state['frame']
            self.inference_memory[0, i].copy_(frame[0, 0])

        # placeholder inference

        # pylint: disable=not-callable
        actions = torch.ones((len(pending_rpcs), 1), dtype=torch.int64).cuda()
        ##########

        self.inference_steps += len(pending_rpcs)

        for i, rpc_tuple in enumerate(pending_rpcs):
            caller_id, state = rpc_tuple

            self.trajectory_store.add_to_entry(caller_id, state)

            f_answers, self.future_answers[caller_id] = self.future_answers[caller_id], Future(
            )
            f_answers.set_result((actions[i].view(1, 1).cpu(),
                                  self.shutdown,
                                  caller_id))

        self.inference_memory.fill_(0.)

    def loop_training(self):
        start = time.time()

        loop_rrefs = []

        for rref in self.actor_rrefs:
            loop_rrefs.append(rref.remote().loop())

        while not (self.shutdown or
                   (self.training_epoch > self.max_epoch > 0) or
                   (self.training_steps > self.max_steps > 0) or
                   (self.training_time > self.max_time > 0)):
            self.batched_training()

        self.shutdown = True
        self.final_time = time.time()-start
        self._cleanup(loop_rrefs)

    def batched_training(self):
        """Trains on sampled, prefetched trajecotries.
        """
        start = time.time()

        trajectories = self.prefetch_rpc.wait()
        self.prefetch_rpc = self.rref.rpc_async().prefetch()

        # placeholder-training
        for trajectory in trajectories:
            self.training_steps += trajectory['current_length']
        ##########

        self.training_time += time.time()-start

    def _cleanup(self, loop_rrefs):
        if all([loop_rref.to_here(timeout=0) for loop_rref in loop_rrefs]):
            for actor_rref in self.actor_rrefs:
                del actor_rref

        # Shutdown and remove queues
        print("Deleting unused trajectories...")
        #del self.trajectory_queue

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
        while True:
            if len(self.trajectory_store.drop_off_queue) >= self.training_batchsize:
                trajectories = [self.trajectory_store.drop_off_queue.popleft()]
                return trajectories
            time.sleep(0.1)

    @staticmethod
    def listdict_to_dictlist(listdict):
        return {k: [dic[k] for dic in listdict] for k in listdict[0]}
