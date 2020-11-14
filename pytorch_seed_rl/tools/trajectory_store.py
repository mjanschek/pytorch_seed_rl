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

# pylint: disable=not-callable, missing-module-docstring, missing-class-docstring, missing-function-docstring, too-many-arguments, arguments-differ

"""The store class can be configured for various data structures.
"""
import copy
from collections import deque

import torch

from ..functional.util import listdict_to_dictlist


class TrajectoryStore():
    """The store class can:
        #. save and load info for a given unique key (hashmap?)
        #. store a predefined data structure (not that flexible :-/)
    """

    def __init__(self, num_keys, zero_obs, device, max_trajectory_length=128):

        self.device = device
        self.zero_obs = {k: v.to(self.device) for k, v in zero_obs.items()}
        # self.zero_obs = zero_obs

        self.max_trajectory_length = max_trajectory_length

        self.trajectory_counter = 0
        self.episode_counter = 0

        self.drop_off_queue = deque()
        self.internal_store = [self._new_trajectory() for _ in range(num_keys)]

    def _new_trajectory(self):
        trajectory = {
            "global_trajectory_id": torch.tensor(self.trajectory_counter, device=self.device),
            "global_episode_id": torch.tensor(self.episode_counter, device=self.device),
            "complete": torch.tensor(False, device=self.device),
            "current_length": torch.tensor(0, device=self.device),
            "states": self._new_states(),
            "metrics": []
        }

        self.trajectory_counter += 1
        self.episode_counter += 1

        return trajectory

    def _new_states(self):
        states = {}
        for key, value in self.zero_obs.items():
            states[key] = value.repeat(
                (self.max_trajectory_length,) + (1,)*(len(value.size())-1))
        return states

    def _reset_trajectory(self, i, complete):
        self.trajectory_counter += 1

        trajectory = self.internal_store[i]
        trajectory['global_trajectory_id'].fill_(self.trajectory_counter)
        trajectory['current_length'].fill_(0)
        trajectory['metrics'] = []
        self._reset_states(trajectory['states'])

        if complete:
            self.episode_counter += 1
            trajectory['global_episode_id'].fill_(self.episode_counter)
            trajectory['complete'].fill_(False)

    @staticmethod
    def _reset_states(states):
        for value in states.values():
            value.fill_(0)

    def add_to_entry(self, i, state, metrics=None):
        assert all(k in self.zero_obs.keys() for k in state.keys())

        state = {k: v.view(self.zero_obs[k].shape)
                 for k, v in state.items()}

        trajectory = self.internal_store[i]
        current_length = trajectory['current_length'].item()
        states = trajectory['states']
        for key, value in state.items():
            # print(key, states[key][current_length].shape, value.shape)
            states[key][current_length].copy_(value[0])

        trajectory['current_length'] += 1
        if state['done'] and state['episode_step'] > 1:
            trajectory["complete"].fill_(True)

        if trajectory["complete"] or (trajectory['current_length'] == self.max_trajectory_length):
            self._drop(trajectory)
            self._reset_trajectory(i, trajectory["complete"])

        trajectory["metrics"].append(metrics)

    def _drop(self, trajectory):
        trajectory["metrics"] = listdict_to_dictlist(trajectory["metrics"])
        for k, v in trajectory["metrics"].items():
            trajectory["metrics"][k] = torch.cat(v)

        t = copy.deepcopy(trajectory)

        self.drop_off_queue.append(t)
