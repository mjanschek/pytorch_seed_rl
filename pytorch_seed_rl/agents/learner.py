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

from ..data_structures.structure import State, Trajectory
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
                 rollout_length=32,
                 total_epochs=10,
                 total_steps=0,
                 save_path=""):
        # spawn A actors with E environments each

        # create stores:
        # RecurrentStateStore: A X E slots for recurrent states (1 per actor)
        # TrajectoryStore: A x E x L slots: A x E Trajectories with max_length L

        # spawn TrajectoryQueue (FIFO)

        # spawn DeviceBuffer (calls TrajectoryQueue till BATCHSIZE trajectories are buffered)
        self.id = rpc.get_worker_info().id
        self.device = torch.device("cuda")
        #self.model = model.to(self.device)

        self.epoch = 0
        print("Agent", str(rank), "spawned.")

        self.actor_rrefs = []
        self_rref = rpc.RRef(self)

        self.shutdown = False
        self.trainingtime = 0

        self.log = []

        self.inference_counter = 0
        self.training_counter = 0

        for i in range(num_learners, num_actors+num_learners):
            print("Spawning actor{}".format(i))
            actor_info = rpc.get_worker_info("actor{}".format(i))
            actor_rref = rpc.remote(actor_info,
                                    Actor,
                                    args=(i, self_rref, env_spawner))
            actor_rref.remote().loop()

            self.actor_rrefs.append(actor_rref)

        actor_names = [rref.owner_name() for rref in self.actor_rrefs]

        self.trajectory_queue = deque(maxlen=100)

        self.trajectory_store = TrajectoryStore(actor_names,
                                                drop_off_queue=self.trajectory_queue)

        self.env_info = env_spawner.env_info

    def infer(self, rpc_id, actor_name, state, metrics=None):
        """Runs inference as rpc.

        Use https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py for batched rpc reference!
        """
        for k, v in state.items():
            state[k] = v.cuda()

        # pylint: disable=not-callable
        action = torch.tensor(self.env_info['action_space'].sample(),
                              dtype=torch.int64)

        # placeholder inference time
        #time.sleep(0.01)

        self.inference_counter += 1

        inference_info = {
            'inference_id': self.inference_counter
        }

        state['inference_info'] = inference_info
        state['metrics'] = metrics

        self.trajectory_store.add_to_entry(actor_name, State(**state))

        return action, self.shutdown

    def train(self):
        """Trains on sampled, prefetched trajecotries.
        """
        start = time.time()
        for _ in range(1000):
            if len(self.trajectory_queue) > 0:
                trajectory = self.trajectory_queue.popleft()
                print("Received trajectory",
                      str(trajectory["global_trajectory_number"]),
                      "lenght of q:", str(len(self.trajectory_queue)))

                # placeholder inference time
                # time.sleep(1)

                self.training_counter += len(trajectory['states'])
            else:
                time.sleep(0.1)
        self.shutdown = True

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

    def prefetch(self):
        """prefetches data from inference thread
        """

        # call TrajectoryQueue, buffer to Device until BATCHSIZE trajectories are gathered
