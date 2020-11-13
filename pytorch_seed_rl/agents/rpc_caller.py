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

"""RPC object that handles communication with an assigned RPC callee.
"""
from torch.distributed import rpc
from torch.futures import Future

from abc import abstractmethod

from .. import agents


class RpcCaller():
    """RPC object that handles communication with an assigned RPC callee.
    """

    def __init__(self, rank, callee_rref):
        # ASSERTIONS
        # check for RpcCallee being inherited by callee_rref
        assert issubclass(callee_rref._get_type(),
                          agents.rpc_callee.RpcCallee)

        self.callee_rref = callee_rref

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.shutdown = False

    def loop(self):
        """Main loop function of an RpcCaller.

        # . Checks in with assigned '~RpcCallee'.
        # . Loops '~_loop()' until '~self.shutdown' is set True.
        # . Checks out with assigned '~RpcCallee'.
        """
        self.callee_rref.rpc_sync().check_in(self.rank)

        while not self.shutdown:
            self._loop()

        self.callee_rref.rpc_sync().check_out(self.rank)
        self.cleanup()

    @abstractmethod
    def _loop(self):
        """Inner loop function of an RpcCaller. Called by '~loop()'

        Must be implemented by child class
        """
        raise NotImplementedError

    def batched_rpc(self, *args, **kwargs) -> Future:
        """Wrap for batched async RPC ran on remote callee.
        """
        return self.callee_rref.rpc_async().batched_process(*args, **kwargs)

    def cleanup(self):
        """Cleans up after main loop is done. Called by '~cleanup()'

        Calls abstract method '~_cleanup()'
        """
        self._cleanup()

    @abstractmethod
    def _cleanup(self):
        """Cleans up after main loop is done. Called by '~cleanup()'

        Must be implemented by child class.
        """
        raise NotImplementedError
