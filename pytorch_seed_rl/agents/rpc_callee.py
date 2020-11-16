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
import gc
import time
from abc import abstractmethod
from collections import deque
from typing import List, Union

import torch.multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

from ..tools.functions import listdict_to_dictlist


class RpcCallee():
    """RPC object that spawns callers and handles batched rpc calls received from spawned callers.During initiation:

    * Spawns :py:attr:`num_callers` instances of :py:class:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller`
    * Invokes their :py:meth:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller.loop()` methods.

    Parameters
    ----------
    rank : `int`
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`)
    num_callees: `int`
        Number of total callees spawned by mother process.
    num_callers: `int`
        Number of total callers to spawn.
    caller_class: Child class of :py:class:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller`
        Class used to spawn callers.
    caller_args: `list`
        Arguments to pass to :py:attr:`caller_class`.
    future_keys: `list`
        Unique identifiers of future answers.
    rpc_batchsize: `int`
        Number of RPCs to gather before processing them as batch.
    """

    def __init__(self,
                 rank: int,
                 num_callees: int = 1,
                 num_callers: int = 1,
                 caller_class: object = None,
                 caller_args=None,
                 future_keys: list = [None],
                 rpc_batchsize: int = 4):

        # ASSERTIONS
        assert num_callees > 0
        assert num_callers > 0

        # caller_class must be given
        assert caller_class is not None

        # Number of future_keys must be equal or greater rpc_batchsize
        assert rpc_batchsize <= len(future_keys)

        # callee_rref is correct subclass
        from ..agents.rpc_caller import RpcCaller
        assert issubclass(caller_class, RpcCaller)

        # ATTRIBUTES
        self.rpc_batchsize = rpc_batchsize
        self.rank = rank

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rref = RRef(self)
        self.lock_batching = mp.Lock()
        self.shutdown = False
        self.t_start = 0.

        # storage
        self.active_callers = {}
        self.caller_rrefs = []
        self.pending_rpcs = deque()

        self.future_answers = {k: Future() for k in future_keys}

        # spawn actors
        self._spawn_callers(caller_class, num_callees,
                            num_callers, *caller_args)

    def _spawn_callers(self,
                       caller_class: object,
                       num_callees: int,
                       num_callers: int,
                       *args):
        """Spawns instances of :py:attr:`caller_class` on RPC workers.

        Parameters
        ----------
        caller_class: Child class of :py:class:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller`
            Class used to spawn callers.
        num_callees: `int`
            Number of total callees spawned by mother process.
        num_callers: `int`
            Number of total callers to spawn.
        *args:
            Arguments to pass to :py:attr:`caller_class`.
        """
        for i in range(num_callers):
            rank = i + num_callees
            callers_info = rpc.get_worker_info("actor%d" % (rank))

            # Store RRef of spawned caller
            self.caller_rrefs.append(rpc.remote(callers_info,
                                                caller_class,
                                                args=(rank, self.rref, *args)))
        print("{} callers spawned, awaiting start.".format(num_callers))

    def _start_callers(self):
        """Calls :py:meth:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller.loop` method
        of all callers in :py:attr:`self.caller_rrefs`.
        """
        for c in self.caller_rrefs:
            c.remote().loop()
        print("Caller loops started.")

    def loop(self):
        """Main loop method of an :py:class:`RpcCallee`.

        #. Loops :py:meth:`_loop()` until :py:attr:`self.shutdown` is set True.
        #. Calls :py:meth:`_cleanup()`
        """
        self.t_start = time.time()

        print("Waiting for start of main loop...")
        while not (self.shutdown):
            self._loop()

        self.shutdown = True
        self._cleanup()

    @async_execution
    def batched_process(self,
                        caller_id: Union[int, str],
                        *args,
                        **kwargs) -> Future:
        """Wraps :py:meth:`_process_batch()` for batched, asynchronous RPC processes.

        This method appends :py:attr:`self.pending_rpcs` with incoming RPCs.
        If `len(self.pending_rpcs)` reached at least :py:attr:`self.rpc_batchsize`,
        :py:meth:`_process_batch()` is called.
        This ultimately wraps the abstract method :py:meth:`process_batch()`.

        Parameters
        ----------
        caller_id : `int` or `str`
            Unique identifier for caller.
        *args:
            Arguments for :py:meth:`process_batch()`
        **kwargs:
            Keyword arguments for :py:meth:`process_batch()`
        """
        # Hook up wait() invoke with Future object that is returned (see torch RPC tutorials)
        f_answer = self.future_answers[caller_id].then(
            lambda f_answers: f_answers.wait())

        # Use batching lock to prohibit concurrent appending of self.pending_rpcs
        with self.lock_batching:
            self.pending_rpcs.append((caller_id, *args, kwargs))

            # if self.rpc_batchsize is reached, pop pending RPCs and start self._process()
            if len(self.pending_rpcs) >= self.rpc_batchsize:
                process_rpcs = [self.pending_rpcs.popleft()
                                for _ in range(self.rpc_batchsize)]

                self._process_batch(process_rpcs)

        return f_answer

    def _process_batch(self, pending_rpcs: List[tuple]):
        """Prepares batched data held by :py:attr:`pending_rpcs` and
        invokes :py:meth:`process_batch()` on this data.
        Sets :py:class:`Future` with according results.

        Parameters
        ----------
        pending_rpcs: `list[tuple]`
            List of tuples that hold data from RPCs.
        """

        caller_ids, *args, kwargs = zip(*pending_rpcs)
        args = [listdict_to_dictlist(b) for b in args]
        kwargs = listdict_to_dictlist(kwargs)

        process_output = self.process_batch(caller_ids, *args, **kwargs)

        for caller_id, result in process_output.items():
            f_answers = self.future_answers[caller_id]
            self.future_answers[caller_id] = Future()
            f_answers.set_result((result,
                                  self.shutdown,
                                  caller_id,
                                  dict()
                                  ))

    def _get_runtime(self) -> float:
        """Returns current runtime in seconds.
        """
        return time.time()-self.t_start

    def check_in(self, caller_id: Union[int, str]):
        """Sets a caller active.

        Should be invoked via RPC by a caller.
        """
        self.active_callers[caller_id] = True

    def check_out(self, caller_id: Union[int, str]):
        """Sets a caller inactive.

        Should be invoked via RPC by a caller.
        """
        del self.active_callers[caller_id]

    @abstractmethod
    def _loop(self):
        """Inner loop function of an :py:class:`RpcCallee`.

        Called by :py:meth:`loop()`.
        """
        raise NotImplementedError

    @abstractmethod
    def process_batch(self,
                      caller_ids: List[Union[int, str]],
                      *args,
                      **kwargs) -> dict:
        """Inner method to process a whole batch at once.

        Called by :py:meth:`_process_batch()`

        Parameters
        ----------
        caller_ids : `list[int]` or `list[str]`
            List of unique identifiers for callers.
        """
        raise NotImplementedError

    @abstractmethod
    def _cleanup(self):
        """Cleans up after main loop is done. Called by :py:meth:`loop()`

        Should invoke super() method, if implemented by child class.
        """
        # Answer pending rpcs to enable actors to terminate
        print("Answering pending RPCs")
        self._answer_rpcs()

        # Run garbage collection to ensure freeing of resources.
        gc.collect()

    def _answer_rpcs(self):
        """Answers all pending RPCs if their caller is not inactive, yet.
        """
        while len(self.active_callers) > 0:

            try:
                rpc_tuple = self.pending_rpcs.popleft()
            except IndexError:
                time.sleep(0.1)
                continue

            caller_id = rpc_tuple[0]

            # return answer=None, shutdown=True, answer_id=caller_id, empty dict
            self.future_answers[caller_id].set_result((None,
                                                       True,
                                                       caller_id,
                                                       dict()))
