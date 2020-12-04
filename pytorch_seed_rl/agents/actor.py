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
#
# pylint: disable=empty-docstring
"""
"""
import time

from torch import tensor
from torch.distributed.rpc import RRef
from torch.futures import Future

from .. import agents
from ..environments import EnvSpawner
from .rpc_caller import RpcCaller


class Actor(RpcCaller):
    """Agent that generates trajectories from at least one environment.

    Sends observations (and metrics) off to inference threads
    on :py:class:`~.agents.Learner`, receives actions.

    Parameters
    ----------
    rank : `int`
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`).
    infer_rref: :py:class:`torch.distributed.rpc.RRef`
        RRef referencing a remote :py:class:`~.agents.Learner`.
    env_spawner: :py:class:`~.EnvSpawner`
        Object that spawns an environment on invoking it's
        :py:meth:`~.EnvSpawner.spawn()` method.
    """

    def __init__(self,
                 rank: int,
                 infer_rref: RRef,
                 env_spawner: EnvSpawner):
        # ASSERTIONS
        # infer_rref must be a Learner
        assert infer_rref._get_type() is agents.Learner

        super().__init__(rank, infer_rref)

        self._num_envs = env_spawner.num_envs
        self._envs = env_spawner.spawn()
        self._current_states = [env.initial() for env in self._envs]

        # pylint: disable=not-callable
        self._metrics = [{'latency': tensor(0.).view(1, 1)}
                         for _ in range(self._num_envs)]

        self._futures = []

    def _loop(self):
        """Inner loop method of an :py:class:`Actor`.
            Called by :py:meth:`~.RpcCaller.loop()`.

            Implements :py:meth:`~.RpcCaller._loop()`.
        """
        self.act()

    def act(self):
        """Interact with internal environment.

            # . Send current state (and metrics) off to batching layer for inference.
            # . Receive action.
        """

        # Send off inference requests for all environments at once, take time
        send_time = time.time()
        self._futures = [self._act(i) for i in range(self._num_envs)]

        # Wait for requested action for each environment.
        if not self.shutdown:
            for i, rpc_tuple in enumerate(self._futures):
                action, self.shutdown, answer_id, inference_infos = rpc_tuple.wait()

                # If requested action is None, Learner was shutdown. Loop can be exited here.
                if action is None:
                    return

                # pylint: disable=not-callable
                self._metrics[i] = {
                    'latency': tensor(time.time() - send_time).view(1, 1)
                }

                # sanity: assert answer is actually for this environment
                assert self._gen_env_id(i) == answer_id

                # perform an environment step,
                # save new state and possible information recorded during inference on the Learner.
                self._current_states[i] = self._envs[i].step(action)
                self._current_states[i] = {
                    **self._current_states[i], **inference_infos}

    def _act(self, i: int) -> Future:
        """Wraps rpc call that is processed batch-wise by a :py:class:`~.agents.Learner`.
            Calls :py:meth:`~.RpcCaller.batched_rpc()`.

            Called by :py:meth:`act()`
        """
        return self.batched_rpc(self._gen_env_id(i),
                                self._current_states[i],
                                metrics=self._metrics[i]
                                )

    def _gen_env_id(self, i: int) -> int:
        """Returns the global environment id, given the local id.
        """
        return (self.rank-1)*self._num_envs+i

    def _cleanup(self):
        """Cleans up after main loop is done.

            Implements :py:meth:`~.RpcCaller._cleanup()`.
        """
        for state in {**self._current_states, **self._metrics}.values():
            del state

        # in case this actor renders an environment
        for env in self._envs:
            env.close()
