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
from torch.optim import RMSprop
from torch.nn import DataParallel

from pytorch_seed_rl.agents import Learner
from pytorch_seed_rl.environments import EnvSpawner
from pytorch_seed_rl.nets import AtariNet
#from pytorch_seed_rl.model import Model

EXPERIMENT_NAME = 'Pong_torchbeast_settings'

ENV_ID = 'PongNoFrameskip-v4'
NUM_ENVS = 1

LEARNER_NAME = "learner{}"
ACTOR_NAME = "actor{}"
TOTAL_EPISODE_STEP = 10000000

NUM_LEARNERS = 1
NUM_ACTORS = 16
CSV_FILE = './csv/'

USE_LSTM = False


def run_threads(rank, world_size, env_spawner, model, optimizer):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if rank < NUM_LEARNERS:
        # rank < NUM_LEARNERS are learners
        rpc.init_rpc(LEARNER_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)

        learner_rref = rpc.remote(LEARNER_NAME.format(rank),
                                  Learner,
                                  args=(rank,
                                        NUM_LEARNERS,
                                        NUM_ACTORS,
                                        env_spawner,
                                        model,
                                        optimizer),
                                  kwargs={'max_steps': TOTAL_EPISODE_STEP,
                                          'exp_name': EXPERIMENT_NAME})

        #learner_rref = rpc.RRef(LEARNER_NAME.format(rank))
        train_rref = learner_rref.remote().loop_training()
        train_rref.to_here(timeout=0)
        # learner_rref.rpc_sync().report()
        # block until all rpcs finish, and shutdown the RPC instance
    else:
        rpc.init_rpc(ACTOR_NAME.format(rank),
                     rank=rank,
                     world_size=world_size)
    rpc.shutdown()


def main():
    # create and wrap environment
    env_spawner = EnvSpawner(ENV_ID, NUM_ENVS)

    # model
    model = AtariNet(
        env_spawner.env_info['observation_space'].shape,
        env_spawner.env_info['action_space'].n,
        USE_LSTM
    )
    model.share_memory()
    #model = DataParallel(model)

    optimizer = RMSprop(
        model.parameters(),
        lr=0.0004,
        momentum=0,
        eps=0.01,
        alpha=0.99
    )

    world_size = NUM_LEARNERS + NUM_ACTORS

    mp.set_start_method('spawn')
    mp.spawn(
        run_threads,
        args=(world_size, env_spawner, model, optimizer),
        nprocs=world_size,
        join=True
    )
    # print("All Processes closed.")


if __name__ == '__main__':
    main()
