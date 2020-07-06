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
import torch.distributed.rpc as rpc


class Actor():
    """Agent that generates trajectories from at least one environment.

    Sends observations (and metrics) off to inference threads on
    :py:class:`~pytorch_seed_rl.agents.learner`, receives actions.
    """

    def __init__(self, rank, learner_rref, env_spawner):
        self.learner_rref = learner_rref
        self.id = rpc.get_worker_info().id
        self.rank = rank
        self.num_envs = env_spawner.num_envs
        self.env = env_spawner.spawn()
        self.current_observation = self.env.reset()

        self._loop()

    def _loop(self, total=100):
        """Loop acting method.
        """
        if total:
            for _ in range(total):
                self.act()
        else:
            while True:
                self.act()

    def act(self):
        """Interact with internal environment.

            #. Send current state (and metrics) off to batching layer for inference.
            #. Receive action.
        """
        ret = self.learner_rref.rpc_async().infer(self.id, self.current_observation)
        action = ret.wait()
        next_obs, rewards, terminals, infos = self.env.step(action)
        self.current_observation = next_obs
