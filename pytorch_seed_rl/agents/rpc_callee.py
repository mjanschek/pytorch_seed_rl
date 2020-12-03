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
from abc import abstractmethod
from collections import deque
from threading import Thread
from typing import List, Union

import torch.multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

from ..tools.functions import listdict_to_dictlist


class RpcCallee():
    """RPC object that spawns callers
    and handles batched rpc calls received from spawned callers.During initiation:

    * Spawns :py:attr:`num_callers` instances of
      :py:class:`~pytorch_seed_rl.agents.rpc_caller.RpcCaller`
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
    """

    def __init__(self,
                 rank: int,
                 num_callees: int = 1,
                 num_callers: int = 1,
                 num_process_threads: int = 1,
                 caller_class: object = None,
                 caller_args=None,
                 future_keys: list = None):

        # ASSERTIONS
        assert num_callees > 0
        assert num_callers > 0

        # caller_class must be given
        assert caller_class is not None

        # callee_rref is correct subclass
        # use import here to omit circular import
        # pylint: disable=import-outside-toplevel
        from ..agents.rpc_caller import RpcCaller
        assert issubclass(caller_class, RpcCaller)
        assert isinstance(future_keys, list)

        # ATTRIBUTES
        self.rank = rank

        # pylint: disable=invalid-name
        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rref = RRef(self)
        self.lock_batching = mp.Lock()
        self.t_start = 0.
        self.shutdown = False
        self._shutdown_done = False

        # counters
        self._loop_iteration = 0

        # storage
        self.active_callers = {}
        self.caller_rrefs = []
        self.pending_rpcs = deque()

        self.future_answers = {k: Future() for k in future_keys}
        self.current_futures = deque(maxlen=len(future_keys))

        # self.inference_thread = self.rref.remote()._process_batch()
        self.processing_threads = [
            Thread(target=self._process_batch,
                   daemon=True,
                   name='processing_thread_%d' % i)
            for i in range(num_process_threads)]

        # spawn actors
        self._spawn_callers(caller_class,
                            num_callees,
                            num_callers,
                            *caller_args)

        for t in self.processing_threads:
            t.start()

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
        for caller_rref in self.caller_rrefs:
            caller_rref.remote().loop()
        print("Caller loops started.")

    def loop(self):
        """Main loop method of an :py:class:`RpcCallee`.

        #. Loops :py:meth:`_loop()` until :py:attr:`self.shutdown` is set True.
        #. Calls :py:meth:`_cleanup()`
        """
        self.t_start = time.time()

        print("Loop started. You can interupt using strg+c.")
        while not self.shutdown:
            self._loop_iteration += 1
            self._loop()

        self._cleanup()
        self._shutdown_done = True

    @async_execution
    def batched_process(self,
                        caller_id: Union[int, str],
                        *args,
                        **kwargs) -> Future:
        """Wraps :py:meth:`_process_batch()` for batched, asynchronous RPC processes.

        This method appends :py:attr:`self.pending_rpcs` with incoming RPCs.

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

        self.current_futures.append((caller_id, f_answer))

        return f_answer

    def _process_batch(self, check_interval: float = 0.001):
        """Prepares batched data held by :py:attr:`self.pending_rpcs` and
        invokes :py:meth:`process_batch()` on this data.
        Sets :py:class:`Future` with according results.

        Parameters
        ----------
        pending_rpcs: `list[tuple]`
            List of tuples that hold data from RPCs.
        """
        while not self.shutdown:
            # check once every microsecond
            time.sleep(check_interval)
            # print(len(self.pending_rpcs))
            with self.lock_batching:
                if len(self.pending_rpcs) == 0:
                    # skip, if no rpcs pending
                    continue
                else:
                    pending_rpcs = [self.pending_rpcs.popleft()
                                    for _ in range(len(self.pending_rpcs))]

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
        print("Answering pending RPCs.")
        self._answer_rpcs()

        print("Joining processing threads.")
        for thread in self.processing_threads:
            try:
                thread.join(timeout=5)
            except RuntimeError:
                # dead thread
                pass

    def _answer_rpcs(self, threshold: int = 10):
        """Answers all pending RPCs if their caller is not inactive, yet.
        """
        tries = 0
        while tries < threshold:
            try:
                rpc_tuple = self.pending_rpcs.popleft()
                tries = 0
            except IndexError:
                tries += 1
                time.sleep(1)
                continue

            caller_id = rpc_tuple[0]
            self._answer_future(self.future_answers[caller_id], caller_id)

        # sometimes, futures are missing from pending_rpcs
        for _ in range(len(self.current_futures)):
            caller_id, f_answer = self.current_futures.popleft()
            try:
                self._answer_future(f_answer, caller_id)
            except RuntimeError:  # already answered
                pass

    def _answer_future(self, future, caller_id):
        future.set_result((None,
                           True,
                           caller_id,
                           dict()))

    def set_shutdown(self):
        """Sets `shutdown` True.
        """
        self.shutdown = True

    def get_shutdown(self):
        """Sets `shutdown` True.
        """
        return self.shutdown

    def get_shutdown_done(self):
        """Sets `shutdown` True.
        """
        return self._shutdown_done
