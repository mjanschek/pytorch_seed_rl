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

import torch.distributed.rpc as rpc


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
        observation = self.env.reset()
        self.current_state = {
            "observation": observation,
            "reward": 0.,
            "terminal": False
        }

        self.shutdown = False
        self.rpc_id = 0
        self._gen_rpc_id()

    def loop(self):
        """Loop acting method.
        """
        while not self.shutdown:
            self.act()
        gc.collect()

    def _act(self):
        """Wrap for async RPC method infer() ran on remote learner.
        """
        return self.infer_rref.rpc_async().infer(self._next_rpc_id(),
                                                 self.name,
                                                 self.current_state['observation'],
                                                 self.current_state['reward'],
                                                 self.current_state['terminal'])

    def act(self):
        """Interact with internal environment.

            #. Send current state (and metrics) off to batching layer for inference.
            #. Receive action.
        """
        pending_action = self._act()
        action, self.shutdown = pending_action.wait()

        observation, reward, terminal, _ = self.env.step(action)
        self.current_state = {
            "observation": observation,
            "reward": reward,
            "terminal": terminal
        }

    def _next_rpc_id(self):
        self.rpc_id += 1
        return self._gen_rpc_id()

    def _gen_rpc_id(self):
        self.current_rpc_id = self.name + "-" + str(self.rpc_id)
        return self.current_rpc_id
