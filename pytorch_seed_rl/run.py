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
import argparse
import json
import os
import shutil

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.optim import Adam, RMSprop

from .agents import Learner
from .environments import EnvSpawner
from .nets import AtariNet

parser = argparse.ArgumentParser(description="PyTorch_SEED_RL")

# basic settings
parser.add_argument("name", default="",
                    help="Experiments name, defaults to environment id.")
parser.add_argument('-R', '--reset',
                    help='USE WITH CAUTION!\n' +
                    'Resets existing experiment, this removes all data on subdir level.',
                    action='store_true')
parser.add_argument('-v', '--verbose',
                    help='Prints system metrics to command line.' +
                    'Set --print_interval for number of training epochs between prints.',
                    action='store_true')
parser.add_argument('--gpu_ids', default="", type=str,
                    help='A comma-separated list of cuda ids this program is permitted to use.')
parser.add_argument('--print_interval', default=10, type=int,
                    help='Number of training epochs between prints.')
parser.add_argument("--savedir", default=os.path.join(os.environ.get("HOME"),
                                                      'logs',
                                                      'pytorch_seed_rl'),
                    type=str, help="Root dir where experiment data will be saved.")
parser.add_argument('--render',
                    action='store_true')

# General training settings
parser.add_argument("--total_steps", default=100000, type=int,
                    help="Total environment steps.")
parser.add_argument("--max_epoch", default=-1, type=int,
                    help="Training epoch limit. Set to -1 for no limit.")
parser.add_argument("--max_time", default=-1, type=float,
                    help="Runtime limit. Set to -1 for no limit.")
parser.add_argument("--batchsize_training", default=4, type=int,
                    help="Training batch size.")
parser.add_argument("--rollout", default=80, type=int,
                    help="The rollout length used for training. \n" +
                    "See IMPALA paper for more info.")
parser.add_argument("--batchsize_inference", default=8, type=int,
                    help="Inference batch size.")
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")

# Environment settings
parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4",
                    help="Gym environment.")
parser.add_argument("--num_env", type=int, default=8,
                    help="Number of environments per actor.")

# Architecture settings
parser.add_argument("--master_address", default='localhost', type=str,
                    help="The master adress for the RPC processgroup. \n" +
                    "WARNING: CHANGE WITH CAUTION!")
parser.add_argument("--master_port", default='29500', type=str,
                    help="The master port for the RPC processgroup. \n" +
                    "WARNING: CHANGE WITH CAUTION!")
parser.add_argument("--num_actors", default=2, type=int,
                    help="Number of actors.")
parser.add_argument("--num_prefetchers", default=1, type=int,
                    help="Number of prefetch processes.")
parser.add_argument('--tensorpipe',
                    help='Uses the default RPC backend of pytorch, Tensorpipe.',
                    action='store_true')

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--optimizer", default='rmsprop',
                    type=str, help="Optimizer used for weight updates.")
parser.add_argument("--learning_rate", default=0.0006,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--epsilon", default=1e-08, type=float,
                    help="Optimizer epsilon for numerical stability.")
parser.add_argument("--decay", default=0, type=float,
                    help="Optimizer weight decay.")

# Adam specific settings.
parser.add_argument("--beta_1", default=0.9, type=float,
                    help="Adam beta 1.")
parser.add_argument("--beta_2", default=0.999, type=float,
                    help="Adam beta 2.")

# RMSProp specific settings.
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")

LEARNER_NAME = "learner{}"
ACTOR_NAME = "actor{}"


def _run_threads(rank,
                 world_size,
                 env_spawner,
                 model,
                 optimizer,
                 flags):
    """Initializes RPC clients.

    Intended use as target function for :py:func:`torch.multiprocessing.spawn()`.

    * Spawns a :py:class:`~pytorch_seed_rl.agents.Learner` as client with rank 0.
    * Spawns :py:class:`~pytorch_seed_rl.agents.Actor` as client with rank greater than 0.

    Parameters
    ----------
    rank: `int`
        The rank of the client within the multiprocessing Processgroup.
    worldsize: `int`
        The total number of clients within the multiprocessing Processgroup.
    env_spawner : :py:class:`~pytorch_seed_rl.environments.env_spawner.EnvSpawner`
        Object that spawns an environment on invoking it's
        :py:meth:`~pytorch_seed_rl.environments.env_spawner.EnvSpawner.spawn()` method.
    model : :py:class:`torch.nn.Module`
        A torch model that processes frames as returned
        by an environment spawned by :py:attr:`env_spawner`
    optimizer : :py:class:`torch.nn.Module`
        A torch optimizer that links to :py:attr:`model`
    """
    os.environ['MASTER_ADDR'] = flags.master_address
    os.environ['MASTER_PORT'] = flags.master_port

    if flags.tensorpipe:
        backend = rpc.BackendType.TENSORPIPE
    else:
        backend = rpc.BackendType.PROCESS_GROUP

    if rank == 0:

        rpc.init_rpc(LEARNER_NAME.format(rank),
                     backend=backend,
                     rank=rank,
                     world_size=world_size,
                     )

        learner_rref = rpc.remote(LEARNER_NAME.format(rank),
                                  Learner,
                                  args=(rank,
                                        1,
                                        flags.num_actors,
                                        env_spawner,
                                        model,
                                        optimizer),
                                  kwargs={'total_steps': flags.total_steps,
                                          'exp_name': flags.name,
                                          'save_path': flags.savedir,
                                          'inference_batchsize': flags.batchsize_inference,
                                          'training_batchsize': flags.batchsize_training,
                                          'rollout_length': flags.rollout,
                                          'max_epoch': flags.max_epoch,
                                          'max_time': flags.max_time,
                                          'num_prefetchers': flags.num_prefetchers,
                                          'verbose': flags.verbose,
                                          'print_interval': flags.print_interval,
                                          'render' : flags.render
                                          })

        training_rref = learner_rref.remote().loop()
        training_rref.to_here(timeout=0)
    else:
        rpc.init_rpc(ACTOR_NAME.format(rank),
                     backend=backend,
                     rank=rank,
                     world_size=world_size,
                     )

    # block until all rpcs finish
    rpc.shutdown()


def _write_flags(flags):
    """Saves flags as a json. Creates directories if needed.

    This function expects `full_path` and `json_path` to be present in `flags`.

    Parameters
    ----------
    flags: :py:obj:`Namespace`
        Namespace object that contains this experiments config.
    """
    os.makedirs(flags.full_path, exist_ok=True)

    with open(flags.json_path, 'w', encoding='utf-8') as f:
        json.dump(vars(flags), f, ensure_ascii=False, indent=4)


def main(flags):
    """
    """
    if flags.name == "":
        flags.name = flags.env

    flags.full_path = os.path.join(flags.savedir, flags.name)
    flags.json_path = os.path.join(flags.full_path, 'config.json')

    if os.path.isfile(flags.json_path) and not flags.reset:
        print("EXPERIMENT DATA EXISTS. CHANGE --savedir OR RESTART WITH RESET FLAG (-R) TO OVERWRITE DATA!")
        return

    shutil.rmtree(flags.full_path, ignore_errors=True)
    _write_flags(flags)

    # create and wrap environment
    env_spawner = EnvSpawner(flags.env, flags.num_env)

    # model
    model = AtariNet(
        env_spawner.env_info['observation_space'].shape,
        env_spawner.env_info['action_space'].n,
        flags.use_lstm
    )

    optim_map = {'adam': lambda parameters: Adam(parameters,
                                                 lr=flags.learning_rate,
                                                 betas=(flags.b_1, flags.b_2),
                                                 eps=flags.epsilon,
                                                 weight_decay=flags.decay),
                 'rmsprop': lambda parameters: RMSprop(parameters,
                                                       lr=flags.learning_rate,
                                                       alpha=flags.alpha,
                                                       momentum=flags.momentum,
                                                       eps=flags.epsilon,
                                                       weight_decay=flags.decay),
                 }

    optimizer = optim_map[flags.optimizer](model.parameters())
    world_size = 1 + flags.num_actors

    mp.set_start_method('spawn')
    mp.spawn(
        _run_threads,
        args=(world_size,
              env_spawner,
              model,
              optimizer,
              flags),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    if FLAGS.gpu_ids != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids
    main(FLAGS)
