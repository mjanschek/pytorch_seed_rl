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


class EnvSpawner():
    """Class that is given to actor threads to spawn local environments.
    """

    def __init__(self, env_id, num_envs=1):
        self.env_id = env_id
        self.num_envs = num_envs

    def spawn(self):
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(self.env_id),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )
