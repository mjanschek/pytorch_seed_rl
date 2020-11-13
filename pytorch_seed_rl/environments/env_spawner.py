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

"""Class that is given to actor threads to spawn local environments.
"""

from . import atari_wrappers
from .environment import Environment


class EnvSpawner():
    """Class that is given to actor threads to spawn local environments.
    """

    def __init__(self, env_id, num_envs=1):
        self.env_id = env_id
        self.num_envs = num_envs
        self.env_info, self.placeholder_obs = self._generate_env_info()

    def _generate_env_info(self):
        """Spawn environment once to save properties for later reference by learner and model
        """
        placeholder_env = self.spawn()[0]

        env_info = {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
            "action_space": placeholder_env.gym_env.action_space,
            "observation_space": placeholder_env.gym_env.observation_space,
            "reward_range": placeholder_env.gym_env.reward_range,
            "max_episode_steps": placeholder_env.gym_env.spec.max_episode_steps
        }

        placeholder_obs = placeholder_env.initial()

        placeholder_env.close()
        del placeholder_env

        return env_info, placeholder_obs

    def spawn(self):
        return [Environment(
            atari_wrappers.wrap_pytorch(
                atari_wrappers.wrap_deepmind(
                    atari_wrappers.make_atari(self.env_id),
                    clip_rewards=False,
                    frame_stack=True,
                    scale=False,
                )
            )
        ) for _ in range(self.num_envs)]
