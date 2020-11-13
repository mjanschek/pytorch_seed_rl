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
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`)
    callee_rref: :py:class:`torch.distributed.rpc.RRef`
        RRef referencing a remote :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
    """

    def __init__(self, rank, callee_rref):
        # ASSERTIONS
        # check for RpcCallee being inherited by callee_rref
        from ..agents.rpc_callee import RpcCallee
        assert issubclass(callee_rref._get_type(), RpcCallee)

        # ATTRIBUTES
        self.callee_rref = callee_rref
        self.rank = rank
        self.shutdown = False

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name

    def loop(self):
        """Main loop function of an RpcCaller.

        #. Checks in with assigned :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
        #. Loops :py:meth:`_loop()` until :py:attr:`self.shutdown` is set True.
        #. Checks out with assigned :py:class:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee`.
        """
        self.callee_rref.rpc_sync().check_in(self.rank)

        while not self.shutdown:
            self._loop()

        self.callee_rref.rpc_sync().check_out(self.rank)
        self._cleanup()

    def batched_rpc(self, *args, **kwargs) -> Future:
        """Wrap for batched async RPC ran on remote callee.
        """
        return self.callee_rref.rpc_async().batched_process(*args, **kwargs)

    @abstractmethod
    def _loop(self):
        """Inner loop function of an RpcCaller. Called by :py:meth:`loop()`
        """
        raise NotImplementedError

    @abstractmethod
    def _cleanup(self):
        """Cleans up after main loop is done. Called by :py:meth:`loop()`
        """
        raise NotImplementedError
