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
from abc import abstractmethod

from torch.distributed import rpc
from torch.futures import Future


class RpcCaller():
    """RPC object that handles communication with an assigned RPC callee.

    Parameters
    ----------
    rank : `int`
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`).
    callee_rref: :py:class:`torch.distributed.rpc.RRef`
        RRef referencing a remote :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
    """

    def __init__(self, rank: int, callee_rref: rpc.RRef):
        # ASSERTIONS
        # check for RpcCallee being inherited by callee_rref
        # use import here to omit circular import
        # pylint: disable=import-outside-toplevel
        from ..agents.rpc_callee import RpcCallee
        assert issubclass(callee_rref._get_type(), RpcCallee)

        # ATTRIBUTES
        self.callee_rref = callee_rref
        self.rank = rank
        self._loop_iteration = 0

        # pylint: disable=invalid-name
        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.shutdown = False
        self._shutdown = False

    def loop(self):
        """Main loop function of an :py:class:`RpcCaller`.

        #. Checks in with assigned :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
        #. Loops :py:meth:`_loop()` until :py:attr:`self.shutdown` is set True.
        #. Checks out with assigned :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
        #. Calls :py:meth:`_cleanup()`
        """
        while not self.shutdown:
            self._loop_iteration += 1
            self._loop()

        self._cleanup()
        self._shutdown = True

    def batched_rpc(self, *args, **kwargs) -> Future:
        """Wrap for batched async RPC ran on remote callee.
        """
        future_action = self.callee_rref.rpc_async().batched_process(*args, **kwargs)
        return future_action

    @abstractmethod
    def _loop(self):
        """Inner loop method of an :py:class:`RpcCaller`. Called by :py:meth:`loop()`
        """
        raise NotImplementedError

    @abstractmethod
    def _cleanup(self):
        """Cleans up after main loop is done. Called by :py:meth:`loop()`
        """
        raise NotImplementedError

    def get_shutdown(self):
        """Sets `shutdown` True.
        """
        return self._shutdown
