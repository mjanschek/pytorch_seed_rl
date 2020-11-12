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

"""RPC object that is able to handle batched rpc calls.
"""
import gc
import time
from abc import abstractmethod
from collections import deque

import torch.multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from torch.distributed.rpc.functions import async_execution
from torch.futures import Future

from ..functional.util import listdict_to_dictlist


class RpcCallee():
    """RPC object that is able to handle batched rpc calls.
    """

    def __init__(self,
                 rank,
                 num_callees=1,
                 num_callers=1,
                 caller_class=None,
                 caller_args=[],
                 rpc_batchsize=4,
                 max_pending_rpcs=4):

        # save arguments as attributes where needed
        self.rpc_batchsize = rpc_batchsize
        self.max_pending_rpcs = max_pending_rpcs

        assert caller_class is not None
        assert rpc_batchsize <= self.max_pending_rpcs

        self.t_start = 0.

        # rpc stuff
        self.active_envs = {}
        self.shutdown = False
        self.rref = RRef(self)
        self.caller_rrefs = []

        self.id = rpc.get_worker_info().id
        self.name = rpc.get_worker_info().name
        self.rank = rank

        self.batching_lock = mp.Lock()
        self.pending_rpcs = deque(maxlen=self.max_pending_rpcs)
        self.future_answers = [Future() for i in range(self.max_pending_rpcs)]

        # spawn actors
        self._spawn_callers(caller_class, num_callees,
                            num_callers, *caller_args)

    def _spawn_callers(self, caller_class, num_callees, num_callers, *args):
        for i in range(num_callers):
            callers_info = rpc.get_worker_info("actor%d" % (i+num_callees))
            self.caller_rrefs.append(rpc.remote(callers_info,
                                      caller_class,
                                      args=(i, self.rref, *args)))
        print("{} callers spawned, awaiting start.".format(num_callers))

    def _start_callers(self):
        for c in self.caller_rrefs:
            c.remote().loop()
        print("Caller loops started.")

    def loop(self):
        self.t_start = time.time()

        print("Waiting for start of main loop...")
        while not (self.shutdown):
            self._loop()
        print("Loop done")

        self.shutdown = True
        self._cleanup()

    @abstractmethod
    def _loop(self):
        raise NotImplementedError

    @async_execution
    def batched_process(self, caller_id, *args, **kwargs):
        f_answer = self.future_answers[caller_id].then(
            lambda f_answers: f_answers.wait())

        with self.batching_lock:
            self.pending_rpcs.append((caller_id, *args, kwargs))
            if len(self.pending_rpcs) >= self.rpc_batchsize:
                process_rpcs = [self.pending_rpcs.popleft()
                                for _ in range(self.rpc_batchsize)]

                self._process_batch(process_rpcs)

        return f_answer

    def _process_batch(self, pending_rpcs):

        caller_ids, *batch, misc = zip(*pending_rpcs)
        batch = [listdict_to_dictlist(b) for b in batch]
        misc = listdict_to_dictlist(misc)

        process_output = self.process_batch(caller_ids, *batch, **misc)

        for caller_id, result in process_output.items():
            f_answers = self.future_answers[caller_id]
            self.future_answers[caller_id] = Future()
            f_answers.set_result((result,
                                  self.shutdown,
                                  caller_id,
                                  dict()
                                  ))

    @abstractmethod
    def process_batch(self, caller_ids, *batch, **misc):
        raise NotImplementedError

    def _get_runtime(self):
        return time.time()-self.t_start

    def _answer_rpcs(self):
        while len(self.active_envs) > 0:
            try:
                rpc_tuple = self.pending_rpcs.popleft()
            except IndexError:
                time.sleep(0.1)
                continue
            caller_id = rpc_tuple[0]
            # print(i, caller_id)
            self.future_answers[caller_id].set_result((None,
                                                       True,
                                                       caller_id,
                                                       dict()))

    def _cleanup(self):
        # Answer pending rpcs to enable actors to terminate
        print("Answering pending RPCs")
        self._answer_rpcs()

        # Run garbage collection to ensure freeing of resources.
        gc.collect()

    def check_in(self, caller_id):
        self.active_envs[caller_id] = True

    def check_out(self, caller_id):
        del self.active_envs[caller_id]
