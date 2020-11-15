"""The :py:mod:`agents` module holds all classes that can interact within the created RPC group.

Exposed classes:
    * :py:class:`~actor.Actor` (Child of :py:class:`~rpc_caller.RpcCaller`)
    * :py:class:`~learner.Learner` (Child of :py:class:`~rpc_callee.RpcCallee`)

Parent classes:
    * :py:class:`~rpc_callee.RpcCallee`
    * :py:class:`~rpc_caller.RpcCaller`
"""

from .actor import Actor
from .learner import Learner
