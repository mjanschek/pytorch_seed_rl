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
import argparse
import os

import gym
import torch

from .environments import EnvSpawner
from .nets import AtariNet
from .tools import Recorder

PARSER = argparse.ArgumentParser(description="PyTorch_SEED_RL")

# basic settings
PARSER.add_argument("name", default="",
                    help="Experiments name, defaults to environment id.")
PARSER.add_argument('-v', '--verbose',
                    help='Prints system metrics to command line.' +
                    'Set --print_interval for number of training epochs between prints.',
                    action='store_true')
PARSER.add_argument("--total_steps", default=100, type=int,
                    help="Total environment steps.")
PARSER.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4",
                    help="Gym environment.")
PARSER.add_argument('--print_interval', default=10, type=int,
                    help='Number of training epochs between prints.')
PARSER.add_argument("--savedir", default=os.path.join(os.environ.get("HOME"),
                                                      'logs',
                                                      'pytorch_seed_rl'),
                    type=str, help="Root dir where experiment data will be saved.")
PARSER.add_argument('--render',
                    action='store_true',
                    help="Renders an episode as gif, " +
                    " if the recorded data finished with a new point record.")
PARSER.add_argument('--max_gif_length', default=0, type=int,
                    help="Enforces a maximum gif length." +
                    "Rendering is triggered, if recorded data reaches this volume.")
PARSER.add_argument('--gpu_ids', default="", type=str,
                    help='A comma-separated list of cuda ids this program is permitted to use.')


def _get_model_path(flags) -> bool:
    model_path = os.path.join(flags.full_path, 'model', 'final_model.pt')
    if not os.path.isfile(model_path):
        print("NO MODEL FOUND AT PATH %s!" % model_path)
        return None

    return model_path


def main(flags):
    """Evaluate a model.
    """
    if flags.name == "":
        flags.name = flags.env

    flags.full_path = os.path.join(flags.savedir, flags.name)
    flags.model_path = _get_model_path(flags)
    if flags.model_path is None:
        return

    flags.eval_path = os.path.join(flags.full_path, 'eval')

    # create and wrap environment
    env_spawner = EnvSpawner(flags.env, 1)
    env = env_spawner.spawn()[0]

    # model
    model = AtariNet(
        env_spawner.env_info['observation_space'].shape,
        env_spawner.env_info['action_space'].n
    )

    model.load_state_dict(torch.load(flags.model_path))
    model.eval()

    recorder = Recorder(save_path=flags.eval_path,
                        render=flags.render,
                        max_gif_length=flags.max_gif_length)

    _interaction_loop(flags, model, env, recorder)


def _interaction_loop(flags,
                      model: torch.nn.Module,
                      env: gym.Env,
                      recorder: Recorder):
    """Starts interaction loop

    Parameters
    ----------
    flags: `namespace`
        Flags as read by the argument parser.
    model: :py:obj:`torch.nn.Module`
        PyTorch model to evaluate.
    env: :py:class:`gym.Env`
        An environment as spawned by the :py:class:`~EnvSpawner`.
    recorded: :py:class:`~Recorder`
        A recorder that logs and records data.
    """
    # pylint: disable=protected-access
    steps = 0
    episodes = 0
    state = env.initial()

    episode_data = {'episode_id': episodes,
                    'return': 0,
                    'length': 0
                    }
    while steps < flags.total_steps:
        eval_dict = model(state)[0]
        state = env.step(eval_dict['action'])
        steps += 1

        if flags.verbose:
            print("STEP: %4d RETURN %4d" % (state['episode_step'],
                                            state['episode_return']))

        if flags.render:
            # target shape [H, W]
            frame = state['frame'][0]
            recorder._record_frame(frame, checks=False)

        if state['done']:
            episodes += 1
            recorder.log('episodes', episode_data)

            if flags.render:
                recorder.record_eps_id = episodes
                recorder._record_episode(check_return=False)

        episode_data = {'episode_id': episodes,
                        'return': state['episode_return'],
                        'length': state['episode_step']
                        }


if __name__ == '__main__':
    FLAGS = PARSER.parse_args()
    # every thread gets an own physical thread
    os.environ["OMP_NUM_THREADS"] = "1"
    if FLAGS.gpu_ids != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids
    main(FLAGS)
