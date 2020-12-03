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

# pylint: disable=not-callable, empty-docstring
"""
"""
import copy
from typing import List, Union

import torch
from torch.multiprocessing import Lock, Queue

from .functions import dict_to_device, listdict_to_dictlist
from .recorder import Recorder


class TrajectoryStore():
    """Data storage for trajectories produced by :py:class:`~pytorch_seed_rl.agents.Actor`.

    This object includes:
        * Management of stored trajectories, states are registered with the correct episode.
        * Detection of completed trajectories. These reached a state, where `done` is True,
          or they contain a number of states equal to :py:attr:`trajectory_length`.
        * Drop of completed trajectories. Data is queued into :py:attr:`drop_off_queue`,
          which can be accessed by external logic.
        * Reset of dropped trajectories.
          Allocated memory is re-used for the next episode of this environment.

    Parameters
    ----------
    keys: `list` of `int` or `str`
        The unique keys this store manages.
    zero_obs: `dict`
        A dictionary with the exact shape of data that shall be stored.
    device: `torch.device`
        The :py:obj:`torch.device` this stores data is stored on.
    recorder: :py:class:`~.Recorder`
        A :py:class:`~.Recorder` object that logs and records data.
    trajectory_length: `int`
        The number of states a trajectory shall contain when completed.
    """

    def __init__(self,
                 keys: List[Union[int, str]],
                 zero_obs: dict,
                 device: torch.device,
                 out_queue: Queue,
                 recorder: Recorder,
                 trajectory_length: int = 128):
        # ATTRIBUTES
        self.out_queue = out_queue
        self.device = device
        self.recorder = recorder
        self._trajectory_length = trajectory_length

        self.zero_obs = {k: v.to(self.device) for k, v in zero_obs.items()}
        self.zero_obs['episode_id'] = torch.ones(
            (1, 1), device=self.device) * -1
        self.zero_obs['prev_episode_id'] = torch.ones(
            (1, 1), device=self.device) * -1
        self._reset_states(self.zero_obs)

        # Counters
        self._trajectory_counter = 0
        self._episode_counter = len(keys)

        # Setup storage
        self._internal_store = {k: self._new_trajectory() for k in keys}
        self._episode_id_store = {k: i for i, k in enumerate(keys)}
        self._locks_trajectories = {k: Lock() for k in keys}
        self._lock_episode_counter = Lock()

    def _new_trajectory(self) -> dict:
        """Returns a new, empty trajectory.
        """
        trajectory = {
            "trajectory_id": torch.tensor(self._trajectory_counter, device=self.device),
            "complete": torch.tensor(False, device=self.device),
            "current_length": torch.tensor(0, device=self.device),
            "states": self._new_states(),
            "metrics": list()
        }

        self._trajectory_counter += 1

        return trajectory

    def _new_states(self):
        """Returns a new, empty state dictionary.
        """
        states = {}
        for key, value in self.zero_obs.items():
            states[key] = value.repeat(
                (self._trajectory_length,) + (1,)*(len(value.size())-1))
        return states

    def _reset_trajectory(self, trajectory: dict):
        """Resets a trajectory inplace.

        Parameters
        ----------
        trajectory: `dict`
            A trajectory as produced by :py:class:`pytorch_seed_rl.agents.Actor`
        """
        self._trajectory_counter += 1

        trajectory['trajectory_id'].fill_(self._trajectory_counter)
        trajectory['current_length'].fill_(0)
        trajectory['metrics'] = list()

        self._reset_states(trajectory['states'])

    @staticmethod
    def _reset_states(states):
        """Fills given list of states with 0 inplace.

        Parameters
        ----------
        states: `list`
            A list of states as produced from the interaction with an environment.
        """
        for value in states.values():
            value.fill_(0)
        states['episode_id'].fill_(-1)

    def add_to_entry(self,
                     key: str,
                     state: dict,
                     metrics: dict = None):
        """Fills given list of states with 0 inplace.

        Parameters
        ----------
        key: `str`
            The key of the trajectory this state relates to.
            Usually the environments unique global identifier.
        state: `dict`
            A state as produced from the interaction with an environment.
            This can include values from model evaluation.
        metrics: `dict`
            An optional dictionary containing additional values.
        """
        self._locks_trajectories[key].acquire()

        # all keys must be known to store
        assert all(k in self.zero_obs.keys() for k in state.keys())

        # load known trajectory
        internal_trajectory = self._internal_store[key]
        length = internal_trajectory['current_length'].item()
        internal_states = internal_trajectory['states']

        # all metrics keys must be known to store, if trajectory already has valid data
        # metrics = {k: v.to(self.device) for k, v in metrics.items()}

        dict_to_device(metrics, self.device)
        if length == 0:
            internal_trajectory["metrics"] = [metrics]
        else:
            internal_trajectory["metrics"].append(metrics)

        # transform shape
        state = {k: v.view(self.zero_obs[k].shape)
                 for k, v in state.items()}

        # overwrite known state (expected to be empty)
        try:
            for k, value in state.items():
                assert internal_states[k][length].sum() == 0
                internal_states[k][length].copy_(value[0])
        except AssertionError:
            print("state[%s] under store key %s is not 0!" % (k, key))

        # save old episode_id
        old_eps_id = self._episode_id_store[key]
        if state['done']:
            with self._lock_episode_counter:
                self._episode_counter += 1
                self._episode_id_store[key] = self._episode_counter

        # update info
        internal_states['episode_id'][length].fill_(
            self._episode_id_store[key])
        internal_states['prev_episode_id'][length].fill_(old_eps_id)

        internal_trajectory['current_length'] += 1
        if internal_trajectory['current_length'] == self._trajectory_length:
            self._drop(copy.deepcopy(internal_trajectory))
            self._reset_trajectory(internal_trajectory)

        self._locks_trajectories[key].release()

    def _drop(self,
              trajectory: dict):
        """Copies a trajectory and drops the copy on :py:attr:`self.drop_off_queue`.

        The original trajectory will be reset in place to keep its memory allocated.

        Parameters
        ----------
        trajectory: `dict`
            The trajectory to drop.
        """
        trajectory["metrics"] = listdict_to_dictlist(trajectory["metrics"])

        for k, value in trajectory["metrics"].items():
            if isinstance(value[0], torch.Tensor):
                trajectory["metrics"][k] = torch.cat(value)

        # self.logging_func(trajectory)
        self.recorder.log_trajectory(trajectory)
        try:
            self.out_queue.put(trajectory)
        except AssertionError:  # queue closed
            return
