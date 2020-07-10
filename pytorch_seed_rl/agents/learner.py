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
import gc
import time
from collections import deque

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn.functional as functional

from ..data_structures.trajectory_store import TrajectoryStore
from .actor import Actor


class Learner():
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
                 loss,
                 threads_infer=1,
                 threads_prefetch=1,
                 threads_train=1,
                 envs_per_actor=1,
                 inference_buffer=1,
                 batchsize=32,
                 rollout_length=64,
                 total_epochs=10,
                 total_steps=0,
                 save_path=""):
        # spawn A actors with E environments each

        # create stores:
        # RecurrentStateStore: A X E slots for recurrent states (1 per actor)
        # TrajectoryStore: A x E x L slots: A x E Trajectories with max_length L

        # spawn TrajectoryQueue (FIFO)

        # spawn DeviceBuffer (calls TrajectoryQueue till BATCHSIZE trajectories are buffered)

        self.starttraining_lock = mp.Lock()
        self.starttraining_lock.acquire()

        self.batchsize = batchsize
        self.id = rpc.get_worker_info().id
        self.device = torch.device("cuda")
        #self.model = model.to(self.device)

        print("Agent", str(rank), "spawned.")

        self_rref = rpc.RRef(self)

        self.shutdown = False
        self.epoch = 0
        self.trainingtime = 0
        self.inference_counter = 0
        self.training_counter = 0

        self.env_info = env_spawner.env_info
        self.dummy_obs = env_spawner.dummy_obs
        for key, value in self._create_dummy_inference_info().items():
            self.dummy_obs[key] = value

        self.device_buffer = {}
        for key, value in self.dummy_obs.items():
            self.device_buffer[key] = torch.zeros(
                (rollout_length, batchsize) + value.size(), dtype=value.dtype)

        self.log = []

        self.actor_rrefs = self._spawn_actors(num_learners,
                                              num_actors,
                                              env_spawner,
                                              self_rref)
        actor_names = [rref.owner_name() for rref in self.actor_rrefs]

        self.trajectory_queue = deque(maxlen=batchsize*num_actors)
        self.trajectory_store = TrajectoryStore(actor_names,
                                                self.dummy_obs,
                                                max_trajectory_length=rollout_length,
                                                drop_off_queue=self.trajectory_queue)

        # self_rref.remote().loop_prefetch()

        #self.number_prefetch_threads = 1
        # self.prefetch_treads = [mp.Process(name="prefetch_thread_{}".format(i),
        #                                   target=self.loop_prefetch) for i in range(self.number_prefetch_threads)]

        # for p in self.prefetch_treads:
        #    p.start()

    @staticmethod
    def _spawn_actors(num_learners, num_actors, env_spawner, self_rref):
        actor_rrefs = []

        for i in range(num_learners, num_actors+num_learners):
            print("Spawning actor{}".format(i))
            actor_info = rpc.get_worker_info("actor{}".format(i))
            actor_rref = rpc.remote(actor_info,
                                    Actor,
                                    args=(i, self_rref, env_spawner))
            actor_rref.remote().loop()

            actor_rrefs.append(actor_rref)
        return actor_rrefs

    def infer(self, rpc_id, actor_name, state, metrics=None):
        """Runs inference as rpc.

        Use https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py for batched rpc reference!
        """
        for key, value in state.items():
            state[key] = value.cuda()

        # pylint: disable=not-callable
        action = torch.tensor(self.env_info['action_space'].sample(),
                              dtype=torch.int64).view(1, 1)

        # placeholder inference time
        # time.sleep(0.01)

        self.inference_counter += 1

        inference_info = self._create_dummy_inference_info()
        state = {**state, **inference_info}

        self.trajectory_store.add_to_entry(actor_name, state, metrics)

        return action, self.shutdown

    def train(self):
        """Trains on sampled, prefetched trajecotries.
        """
        start = time.time()
        print("begin training")
        slept_time = 0.
        while not self.shutdown and slept_time < 10.:
            if len(self.trajectory_queue) > 0:
                trajectory = self.trajectory_queue.popleft()
                print(
                    "TID:", trajectory['global_trajectory_number'],
                    "t length:", trajectory['current_length'],
                    "q length:", len(self.trajectory_queue))
                self.training_counter += trajectory['current_length']
                self.shutdown = self.training_counter >= 10000
            else:
                time.sleep(0.1)
                slept_time += 0.1

        self.trainingtime = time.time()-start

        # get batches from device

        self._cleanup()

    def _cleanup(self):
        for rref in self.actor_rrefs:
            del rref

        # Shutdown and remove queues
        print("Deleting unused trajectories...")
        # while not self.trajectory_queue.empty():
        #    trajectory = self.trajectory_queue.get()
        #    del trajectory

        # self.trajectory_queue.close()
        # self.trajectory_queue.join_thread()
        del self.trajectory_queue

        gc.collect()

    def report(self):
        """Reports data to a logging system
        """
        fps = self.inference_counter / self.trainingtime

        print("infered", str(self.inference_counter), "times")
        print("in", str(self.trainingtime), "seconds")
        print("==>", str(fps), "fps")

        fps = self.training_counter / self.trainingtime
        print("trained", str(self.training_counter), "times")
        print("in", str(self.trainingtime), "seconds")
        print("==>", str(fps), "fps")

    def checkpoint(self):
        """Checkpoint the model
        """

    # def loop_prefetch(self):
    #     released = False
    #     while not self.shutdown:
    #         self.prefetch()
    #         if not released:
    #             self.starttraining_lock.release()
    #             released = True

    # def prefetch(self):
    #     """prefetches data from inference thread
    #     """
    #     while len(self.device_buffer) < self.batchsize:
    #         if len(self.trajectory_queue) > 0:
    #             trajectory = self.trajectory_queue.popleft()
    #             # create batch
    #             self.device_buffer.append(trajectory)
    #         else:
    #             time.sleep(0.5)

    #     if len(self.device_buffer) == self.batchsize:
    #         with self.batch_lock:
    #             self.buffered_batch = self.create_batch()
    #     self.device_buffer.clear()

    # def create_batch(self):

    #     batch = {
    #         "global_trajectory_numbers": buffer_dl["global_trajectory_number"],
    #         "frames": frames,
    #         "rewards": rewards,
    #         "dones": dones,
    #         "episode_returns": episode_returns,
    #         "episode_steps": episode_steps,
    #         "last_actions": last_actions,
    #         "inference_infos": inference_infos,
    #         "metrics": metrics,
    #     }

    #     return batch

    @staticmethod
    def listdict_to_dictlist(listdict):
        return {k: [dic[k] for dic in listdict] for k in listdict[0]}

    def _create_dummy_inference_info(self):
        return {
            'behaviour_policy': torch.zeros((1, self.env_info['action_space'].n), dtype=torch.float),
            'behaviour_value': torch.zeros((1, 1), dtype=torch.float)
        }
