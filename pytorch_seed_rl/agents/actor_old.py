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

"""Agent that generates trajectories from at least one environment.

Consists of:
    #. n environments
"""
import gc
import time

import torch
import torch.distributed.rpc as rpc

from .. import agents


class Actor():
    """Agent that generates trajectories from at least one environment.

    Sends observations (and metrics) off to inference threads on
    :py:class:`~pytorch_seed_rl.agents.learner`, receives actions.
    """

    def __init__(self, rank, infer_rref, env_spawner):
        self.infer_rref = infer_rref
        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.num_envs = env_spawner.num_envs
        self.env = env_spawner.spawn()
        self.current_states = self.env.initial()

        self.shutdown = False
        self.rpc_id = 0
        self.i_steps = 0
        self._gen_rpc_id()

        self.inference_action = agents.Learner.batched_inference

    def loop(self):
        """Loop acting method.
        """
        while not self.shutdown:
            self.act()

        self.env.close()
        gc.collect()

    def _act(self):
        """Wrap for async RPC method infer() ran on remote learner.
        """
        pending_action = rpc.rpc_sync(self.infer_rref.owner(),
                                       self.inference_action,
                                       args=(self.infer_rref,
                                             self._next_rpc_id(),
                                             self.name,
                                             self.current_states)
                                       )

        return pending_action

    def act(self):
        """Interact with internal environment.

            #. Send current state (and metrics) off to batching layer for inference.
            #. Receive action.
        """
        action = self._act()
        print(self.name, "received", self.current_rpc_id)

        self.current_states = self.env.step(action)
        self.i_steps += 1

    def _next_rpc_id(self):
        self.rpc_id += 1
        return self._gen_rpc_id()

    def _gen_rpc_id(self):
        self.current_rpc_id = self.name + "-" + str(self.rpc_id)
        return self.current_rpc_id
