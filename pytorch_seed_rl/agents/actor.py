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
import time

from torch import tensor

from .. import agents
from .rpc_caller import RpcCaller


class Actor(RpcCaller):
    """Agent that generates trajectories from at least one environment.

        Sends observations (and metrics) off to inference threads on
        :py:class:`~pytorch_seed_rl.agents.Learner`, receives actions.


        TESTS:
        :py:meth:`_loop()`
        :py:method:`_loop()`
        :py:func:`_loop()`
        :py:function:`_loop()`
    """

    def __init__(self, rank, infer_rref, env_spawner):
        # ASSERTIONS
        # infer_rref must be a Learner
        assert infer_rref._get_type() is agents.Learner

        super().__init__(rank, infer_rref)

        self.num_envs = env_spawner.num_envs
        self.envs = env_spawner.spawn()
        self.current_states = [env.initial() for env in self.envs]

        self.steps_infered = 0

        # pylint: disable=not-callable
        self.metrics = [{'latency': tensor(0.).view(1, 1)}
                        for _ in range(self.num_envs)]

    def _loop(self):
        """Inner loop function of an :py:class:`~pytorch_seed_rl.agents.Actor`.
            Called by :py:method:`~pytorch_seed_rl.agents.RpcCaller.loop()`.

            Inherited from :py:class:`~pytorch_seed_rl.agents.RpcCaller`.
        """
        self.act()

    def _act(self, i):
        """Wraps rpc call that is processed batch-wise by a `~pytorch_seed_rl.agents.Learner`.
            Calls :py:method:`batched_rpc()`.

            Called by :py:method:`~act()`
        """
        return self.batched_rpc(self._gen_env_id(i),
                                self.current_states[i],
                                {'test_data': 'v_data'},
                                metrics=self.metrics[i],
                                test={'t': 'v'})

    def act(self):
        """Interact with internal environment.

            #. Send current state (and metrics) off to batching layer for inference.
            #. Receive action.
        """
        # Send off inference requests for all environments at once, take time
        send_time = time.time()
        future_actions = [self._act(i) for i in range(self.num_envs)]

        # Wait for requested action for each environment.
        for i, rpc_tuple in enumerate(future_actions):
            action, self.shutdown, answer_id, inference_infos = rpc_tuple.wait()

            # If requested action is None, Learner was shutdown. Loop can be exited here.
            if action is None:
                break

            # record metrics
            # pylint: disable=not-callable
            self.metrics[i] = {
                'latency': tensor(time.time() - send_time).view(1, 1)
            }

            # sanity: assert answer is actually for this environment
            assert self._gen_env_id(i) == answer_id

            # perform an environment step and save new state and possible information recorded during inference on the Learner.
            self.current_states[i] = self.envs[i].step(action)
            self.current_states[i] = {
                **self.current_states[i], **inference_infos}

            self.steps_infered += 1

    def _cleanup(self):
        """Cleans up after main loop is done.
            Called by :py:method:`~pytorch_seed_rl.agents.RpcCaller.cleanup()`

            Inherited by :py:class:`~pytorch_seed_rl.agents.RpcCaller`.
        """
        for env in self.envs:
            env.close()

    def _gen_env_id(self, i: int) -> int:
        """Returns the global environment id, given the local id.
        """
        return self.rank*self.num_envs+i
