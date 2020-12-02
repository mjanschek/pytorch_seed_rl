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

# pylint: disable=not-callable
"""
"""
import copy
from typing import List, Union

import torch
from torch.multiprocessing import Lock

from .functions import dict_to_device, listdict_to_dictlist


class TrajectoryStore():
    """Data storage for trajectories produced by :py:class:`~pytorch_seed_rl.agents.Actor`.

    This object includes:
        * Management of stored trajectories, states are registered with the correct episode.
        * Detection of completed trajectories. These reached a state, where `done` is True,
          or they contain a number of states equal to :py:attr:`max_trajectory_length`.
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
    max_trajectory_length: `int`
        The number of states a trajectory shall contain when completed.
    max_queued_drops: `int`
        The maximum number of dropped trajectories waiting in the queue.

    Attributes
    ----------
    drop_off_queue: `deque`
        A queue for dropped (complete) trajectories
    """

    def __init__(self,
                 keys: List[Union[int, str]],
                 zero_obs: dict,
                 device: torch.device,
                 out_queue,
                 logging_func,
                 max_trajectory_length: int = 128):
        # ATTRIBUTES
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.zero_obs = {k: v.to(self.device) for k, v in zero_obs.items()}
        self.zero_obs['episode_id'] = torch.ones(
            (1, 1), device=self.device) * -1
        self.zero_obs['prev_episode_id'] = torch.ones(
            (1, 1), device=self.device) * -1
        self._reset_states(self.zero_obs)

        # Counters
        self.trajectory_counter = 0
        self.episode_counter = len(keys)

        # Setup storage
        self.internal_store = {k: self._new_trajectory() for k in keys}
        self.episode_id_store = {k: i for i, k in enumerate(keys)}
        self.locks_trajectories = {k: Lock() for k in keys}
        self.lock_episode_counter = Lock()

        self.out_queue = out_queue
        self.logging_func = logging_func

    def _new_trajectory(self) -> dict:
        """Returns a new, empty trajectory.
        """
        trajectory = {
            "trajectory_id": torch.tensor(self.trajectory_counter, device=self.device),
            "complete": torch.tensor(False, device=self.device),
            "current_length": torch.tensor(0, device=self.device),
            "states": self._new_states(),
            "metrics": list()
        }

        self.trajectory_counter += 1

        return trajectory

    def _new_states(self):
        """Returns a new, empty state dictionary.
        """
        states = {}
        for k, v in self.zero_obs.items():
            states[k] = v.repeat(
                (self.max_trajectory_length,) + (1,)*(len(v.size())-1))
        return states

    def _reset_trajectory(self, trajectory: dict):
        """Resets a trajectory inplace.

        Parameters
        ----------
        trajectory: `dict`
            A trajectory as produced by :py:class:`pytorch_seed_rl.agents.Actor`
        """
        self.trajectory_counter += 1

        trajectory['trajectory_id'].fill_(self.trajectory_counter)
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
        self.locks_trajectories[key].acquire()

        # all keys must be known to store
        assert all(k in self.zero_obs.keys() for k in state.keys())

        # load known trajectory
        internal_trajectory = self.internal_store[key]
        n = internal_trajectory['current_length'].item()
        internal_states = internal_trajectory['states']

        # all metrics keys must be known to store, if trajectory already has valid data
        # metrics = {k: v.to(self.device) for k, v in metrics.items()}

        dict_to_device(metrics, self.device)
        if n == 0:
            internal_trajectory["metrics"] = [metrics]
        else:
            internal_trajectory["metrics"].append(metrics)

        # transform shape
        state = {k: v.view(self.zero_obs[k].shape)
                 for k, v in state.items()}

        # overwrite known state (expected to be empty)
        try:
            for k, v in state.items():
                assert internal_states[k][n].sum() == 0
                internal_states[k][n].copy_(v[0])
        except AssertionError:
            print("state[%s] under store key %s is not 0!" % (k, key))

        # save old episode_id
        old_eps_id = self.episode_id_store[key]
        if state['done']:
            with self.lock_episode_counter:
                self.episode_counter += 1
                self.episode_id_store[key] = self.episode_counter

        # update info
        internal_states['episode_id'][n].fill_(self.episode_id_store[key])
        internal_states['prev_episode_id'][n].fill_(old_eps_id)

        internal_trajectory['current_length'] += 1
        if (internal_trajectory['current_length'] == self.max_trajectory_length):
            self._drop(copy.deepcopy(internal_trajectory))
            self._reset_trajectory(internal_trajectory)

        self.locks_trajectories[key].release()

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

        for k, v in trajectory["metrics"].items():
            if isinstance(v[0], torch.Tensor):
                trajectory["metrics"][k] = torch.cat(v)

        self.logging_func(trajectory)
        try:
            self.out_queue.put(trajectory)
        except ValueError:  # queue closed
            return
