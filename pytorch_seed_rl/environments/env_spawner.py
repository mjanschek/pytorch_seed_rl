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
from typing import List, Tuple

import gym

from . import atari_wrappers


class EnvSpawner():
    """Class that is given to actor threads to spawn local environments by invoking :py:meth:`spawn()`.

    An instance of this class exposes :py:meth:`spawn()`.

    Parameters
    ----------
    env_id: `str`
        The environments identifier as registered with :py:mod:`gym`.
    num_envs: `int`
        The number of environments :py:meth:`spawn()` returns.

    Attributes
    ----------
    self.env_info: `dict`
        Infos about the spawned environments as dictionary.
    self.placeholder_obs: `dict`
        A dictionary with the same structure as observations return by the spawned environments :py:meth:`~gym.Env.step()` method.
    """

    def __init__(self,
                 env_id: str,
                 num_envs: int = 1):
        self.env_id = env_id
        self.num_envs = num_envs
        self._generate_env_info()

    def spawn(self) -> List[gym.Env]:
        """Returns a list of wrapped environments (using OpenAI's :py:mod:`gym`).

        Applies:
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.ClipRewardEnv`
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.DictObservations`
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.EpisodicLifeEnv`
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.FireResetEnv`, if :py:attr:`env` contains an action with meaning 'FIRE'
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.ImageToPyTorch`
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.MaxAndSkipEnv` (`skip` = 4)
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.NoopResetEnv` (`noop_max` = 30)
            * :py:class:`~pytorch_seed_rl.environments.atari_wrappers.WarpFrame`
        """
        return [atari_wrappers.DictObservations(
            atari_wrappers.wrap_pytorch(
                atari_wrappers.wrap_deepmind(
                    atari_wrappers.make_atari(self.env_id),
                    clip_rewards=False,
                    frame_stack=True,
                    scale=False,
                )
            )
        ) for _ in range(self.num_envs)]

    def _generate_env_info(self):
        """Spawns environment once to save properties for later reference by learner and model
        """
        placeholder_env = self.spawn()[0]

        self.env_info = {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
            "action_space": placeholder_env.env.action_space,
            "observation_space": placeholder_env.env.observation_space,
            "reward_range": placeholder_env.env.reward_range,
            "max_episode_steps": placeholder_env.env.spec.max_episode_steps
        }

        self.placeholder_obs = placeholder_env.initial()

        placeholder_env.close()
        del placeholder_env
