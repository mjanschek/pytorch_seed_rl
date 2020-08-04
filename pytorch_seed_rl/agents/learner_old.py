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
import time
from collections import deque

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn.functional as functional

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
                 threads_infer=1,
                 threads_prefetch=1,
                 threads_train=1,
                 inference_buffer=1,
                 inference_batchsize=4,
                 batchsize=4,
                 rollout_length=8,
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
        self.batch_lock = mp.Lock()
        self.starttraining_lock.acquire()

        self.batchsize = batchsize
        self.inference_batchsize = inference_batchsize
        self.pending_rpcs = []

        self.id = rpc.get_worker_info().id
        self.device = torch.device("cuda")
        # self.model = model.to(self.device)

        print("Agent", str(rank), "spawned.")

        self.rref = rpc.RRef(self)

        self.shutdown = False
        self.epoch = 0
        self.step = 0
        self.timespent = 0
        self.final_time = 0
        self.inference_counter = 0
        self.inference_batch_id = 0
        self.training_counter = 0
        self.steps_per_batch = rollout_length * batchsize
        self.id_seen = []

        self.env_info = env_spawner.env_info
        self.dummy_obs = env_spawner.dummy_obs
        for key, value in self.create_dummy_inference_info().items():
            self.dummy_obs[key] = value

        self.device_buffer = self._prepare_buffer(rollout_length,
                                                  batchsize,
                                                  self.dummy_obs,
                                                  self.device)
        self.prefetching_buffer = self._prepare_buffer(rollout_length,
                                                       batchsize,
                                                       self.dummy_obs,
                                                       torch.device('cpu'))

        self.log = []
        self.lock = mp.Lock()
        self.wait_for_inference = mp.Event()

        self.actor_rrefs = self._spawn_actors(num_learners,
                                              num_actors,
                                              env_spawner)
        actor_names = [rref.owner_name() for rref in self.actor_rrefs]

        self.inference_states = {
            key: copy.deepcopy(self.dummy_obs) for key in actor_names}

        self.pending_actions = {
            key: self.dummy_obs['last_action'].clone() for key in actor_names}
        # self.future_actions = {
        #     key: torch.futures.Future() for key in key_list}

        self.future_actions = torch.futures.Future()

        self.trajectory_queue = deque(maxlen=batchsize*num_actors)
        self.trajectory_store = TrajectoryStore(actor_names,
                                                self.dummy_obs,
                                                self.trajectory_queue,
                                                self.device,
                                                max_trajectory_length=rollout_length)

        self.prefetch_rpc = self.rref.rpc_async().prefetch()

    @staticmethod
    def _prepare_buffer(rollout_length, batchsize, dummy_obs, device):
        buffer = {}
        buffer['states'] = {}
        buffer['global_trajectory_number'] = torch.zeros(batchsize,
                                                         device=device)
        buffer['complete'] = torch.zeros(batchsize,
                                         dtype=torch.bool,
                                         device=device)
        buffer['current_length'] = torch.zeros(batchsize,
                                               device=device)
        # buffer['metrics'] = [None]*batchsize

        for key, value in dummy_obs.items():
            buffer['states'][key] = value.repeat(
                (rollout_length, batchsize) + (1,)*(len(value.size())-2)).to(device)
        return buffer

    def _spawn_actors(self, num_learners, num_actors, env_spawner):
        actor_rrefs = []

        for i in range(num_learners, num_actors+num_learners):
            print("Spawning actor{}".format(i))
            actor_info = rpc.get_worker_info("actor{}".format(i))
            actor_rref = rpc.remote(actor_info,
                                    agents.Actor,
                                    args=(i, self.rref, env_spawner))
            actor_rref.remote().loop()

            actor_rrefs.append(actor_rref)
        return actor_rrefs

    @staticmethod
    @rpc.functions.async_execution
    def batched_inference(agent_rref, rpc_id, key, state, metrics=None):
        # see
        # https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py

        self = agent_rref.local_value()

        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[rpc_id]
        )

        #future_action = self.future_actions

        with self.lock:
            print("I:", key, rpc_id, "copy state to buffer")
            for k, v in state.items():
                self.inference_states[key][k].copy_(v)
            self.pending_rpcs.append((key, rpc_id))
            if len(self.pending_rpcs) == self.inference_batchsize:
                print("I:", self.pending_rpcs, "start")

                # placeholder inference
                time.sleep(0.05)
                self.inference_counter += 1

                # pylint: disable=not-callable
                actions = {rpc_id: torch.tensor(self.env_info['action_space'].sample(),
                                                  dtype=torch.int32).view(1, 1)
                           for _, rpc_id in self.pending_rpcs}

                inference_info = self.create_dummy_inference_info()
                ##########
                for done_key, done_rpc in self.pending_rpcs:
                    done_state = self.inference_states[done_key]
                    done_state = {**done_state, **inference_info}

                    self.trajectory_store.add_to_entry(done_key,
                                                       done_state,
                                                       metrics)
                    print("I:", done_key, done_rpc, "done: action", actions[done_rpc].item())

                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions)

                self.pending_rpcs = []

        return future_action

    def infer(self, rpc_id, key, state, metrics=None):
        """Runs inference as rpc.

        Use https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py for batched rpc reference!
        """
        # pylint: disable=not-callable
        action = torch.tensor(self.env_info['action_space'].sample(),
                              dtype=torch.int64).view(1, 1)

        # inference_time
        time.sleep(0.05)
        self.inference_counter += 1

        inference_info = self.create_dummy_inference_info()

        state = {**state, **inference_info}

        self.trajectory_store.add_to_entry(key,
                                           state,
                                           metrics)

        future_actions = torch.futures.Future()
        future_action = future_actions.then(
            lambda future_actions: future_actions.wait().item()
        )
        future_actions.set_result(action)

        return future_action

    def loop_train(self, max_epoch=-1, max_steps=10000, max_time=5):
        start = time.time()

        while not (self.shutdown or
                   (self.epoch > max_epoch > 0) or
                   (self.step > max_steps > 0) or
                   (self.timespent > max_time > 0)):
            self.train()
        self.shutdown = True
        self.final_time = time.time()-start
        self._cleanup()

        #assert len(self.id_seen) == len(set(self.id_seen))

    def train(self):
        """Trains on sampled, prefetched trajecotries.
        """
        start = time.time()

        print("waiting on prefetch...")
        self.prefetch_rpc.wait()
        batch = self.device_buffer
        self.prefetch_rpc = self.rref.rpc_async().prefetch()
        print("TID    :", batch['global_trajectory_number'])
        #print("lengths:", batch['current_length'])

        # training_time
        time.sleep(0.5)
        self.step += sum(batch['current_length'])
        self.id_seen.extend(batch['global_trajectory_number'].tolist())
        assert sum(batch['current_length']) <= self.steps_per_batch
        self.timespent += time.time()-start

    def _cleanup(self):
        for rref in self.actor_rrefs:
            del rref

        # Shutdown and remove queues
        print("Deleting unused trajectories...")
        # while not self.trajectory_queue.empty():
        #    trajectory = self.trajectory_queue.get()
        #    del trajectory

        del self.trajectory_queue

        gc.collect()

    def report(self):
        """Reports data to a logging system
        """
        if self.timespent > 0:
            fps = self.inference_counter / self.timespent

            print("infered", str(self.inference_counter), "times")
            print("in", str(self.timespent), "seconds")
            print("==>", str(fps), "fps")

            fps = self.step / self.timespent
            print("trained", str(self.step), "times")
            print("in", str(self.timespent), "seconds")
            print("==>", str(fps), "fps")

    def checkpoint(self):
        """Checkpoint the model
        """

    def prefetch(self):
        """prefetches data from inference thread
        """
        trajectory_batch_id = 0
        #ids = []
        while trajectory_batch_id < self.batchsize:
            if len(self.trajectory_queue) > 0:
                trajectory = self.trajectory_queue.popleft()
                print("L: Found complete trajectory %d" %
                      trajectory['global_trajectory_number'])

                # ids.append(trajectory['global_trajectory_number'].tolist())
                print("L: Adding to Buffer")
                self._add_to_buffer(self.prefetching_buffer,
                                    trajectory_batch_id,
                                    trajectory)
                trajectory_batch_id += 1
            else:
                time.sleep(0.1)

        print("L: Copy into devicebuffer")
        self._copy_into(self.prefetching_buffer, self.device_buffer)

    @staticmethod
    def _add_to_buffer(buffer, trajectory_batch_id, trajectory):
        for key, value in trajectory['states'].items():
            buffer['states'][key][:, trajectory_batch_id].copy_(value[:, 0])

        for key, value in trajectory.items():
            if key != 'states':
                buffer[key][trajectory_batch_id].copy_(value)

    @staticmethod
    def _copy_into(source_buffer, target_buffer):

        for key, value in source_buffer['states'].items():
            target_buffer['states'][key].copy_(value, non_blocking=True)

        for key, value in source_buffer.items():
            if key != 'states':
                target_buffer[key].copy_(value, non_blocking=True)

    @staticmethod
    def listdict_to_dictlist(listdict):
        return {k: [dic[k] for dic in listdict] for k in listdict[0]}

    def create_dummy_inference_info(self):
        return {
            'behaviour_policy': torch.zeros((1, 1, self.env_info['action_space'].n),
                                            dtype=torch.float,
                                            ),
            'behaviour_value': torch.zeros((1, 1),
                                           dtype=torch.float,
                                           )
        }
