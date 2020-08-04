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

from torch import tensor
from torch.distributed import rpc

from .. import agents


class Actor():
    """Agent that generates trajectories from at least one environment.

    Sends observations (and metrics) off to inference threads on
    :py:class:`~pytorch_seed_rl.agents.learner`, receives actions.
    """

    def __init__(self, rank, infer_rref, env_spawner):
        self.infer_rref = infer_rref
        self.inference_method = agents.Learner.batched_inference

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.num_envs = env_spawner.num_envs
        self.envs = env_spawner.spawn()
        self.current_states = [env.initial() for env in self.envs]

        self.steps_infered = 0

        # pylint: disable=not-callable
        self.metrics = [{'latency': tensor(0.).view(1,1)} for _ in range(self.num_envs)]

        self.shutdown = False

    def loop(self):
        """Loop acting method.
        """
        for i in range(self.num_envs):
            self.infer_rref.rpc_sync().check_in(self._gen_env_id(i))

        while not self.shutdown:
            self.act()

        for i in range(self.num_envs):
            self.infer_rref.rpc_sync().check_out(self._gen_env_id(i))

        for env in self.envs:
            env.close()

        return True

    def _act(self, i):
        """Wrap for async RPC method infer() ran on remote learner.
        """
        future_action = rpc.rpc_async(self.infer_rref.owner(),
                                      self.inference_method,
                                      args=(self.infer_rref,
                                            self._gen_env_id(i),
                                            self.current_states[i],
                                            self.metrics[i]),
                                      timeout=60)

        return future_action

    def act(self):
        """Interact with internal environment.

            #. Send current state (and metrics) off to batching layer for inference.
            #. Receive action.
        """
        send_time = time.time()
        future_actions = [self._act(i) for i in range(self.num_envs)]

        for i, rpc_tuple in enumerate(future_actions):
            action, self.shutdown, answer_id, inference_infos = rpc_tuple.wait()

            latency = time.time() - send_time

            # sanity check
            assert self._gen_env_id(i) == answer_id

            self.current_states[i] = self.envs[i].step(action)
            self.current_states[i] = {
                **self.current_states[i], **inference_infos}

            # pylint: disable=not-callable
            self.metrics[i] = {
                'latency': tensor(latency).view(1,1)
            }
        
        self.steps_infered += self.num_envs

    def _gen_env_id(self, i):
        return self.rank*self.num_envs+i
