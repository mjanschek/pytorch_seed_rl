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

"""Main python script
"""

import os

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

from pytorch_seed_rl.agents import Learner
from pytorch_seed_rl.environments import EnvSpawner
#from pytorch_seed_rl.model import Model

ENV_ID = 'BreakoutNoFrameskip-v4'

LEARNER_NAME = "learner{}"
ACTOR_NAME = "actor{}"
TOTAL_EPISODE_STEP = 100

NUM_ENVS = 1
NUM_LEARNERS = 1
NUM_ACTORS = 3
CSV_FILE = './csv/'


def run_threads(rank, world_size, env_spawner, model, optimizer, loss):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if rank < NUM_LEARNERS:
        # rank < NUM_LEARNERS are learners
        rpc.init_rpc(LEARNER_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)

        rpc.remote(LEARNER_NAME.format(rank),
                   Learner,
                   args=(rank,
                         NUM_LEARNERS,
                         NUM_ACTORS,
                         env_spawner,
                         model,
                         optimizer,
                         loss))
    else:
        rpc.init_rpc(ACTOR_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)
    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()


def main():
    # create and wrap environment
    env_spawner = EnvSpawner(ENV_ID, NUM_ENVS)

    # model
    model = None
    # model.share_memory()

    optimizer = None
    loss = None

    world_size = NUM_LEARNERS + NUM_ACTORS

    mp.spawn(
        run_threads,
        args=(world_size, env_spawner, model, optimizer, loss),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
