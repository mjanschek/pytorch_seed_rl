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

"""The store class can be configured for various data structures.
"""

#from .structure import Trajectory


class TrajectoryStore():
    """The store class can:
        #. save and load info for a given unique key (hashmap?)
        #. store a predefined data structure (not that flexible :-/)
    """

    def __init__(self, key_list, max_trajectory_length=-1, drop_off_queue=None):

        self.max_trajectory_length = max_trajectory_length
        self.drop_off_queue = drop_off_queue
        self.trajectory_counter = 0

        self.internal_store = {key: self._new_trajectory() for key in key_list}

    def _new_trajectory(self):
        trajectory = {
            "global_trajectory_number": self.trajectory_counter,
            "complete": False,
            "states": []
        }

        # Trajectory(global_trajectory_number=self.trajectory_counter,
        #                        complete=False,
        #                        states=[])

        self.trajectory_counter += 1

        return trajectory

    def add_to_entry(self, key, state):
        trajectory = self.internal_store[key]

        trajectory["states"].append(state)

        if state.done:
            trajectory["complete"] = True

        if trajectory["complete"] or (len(trajectory["states"]) == self.max_trajectory_length):
            self._drop(trajectory)
            trajectory = self._new_trajectory()

        self.internal_store[key] = trajectory

    def _drop(self, trajectory):
        self.drop_off_queue.append(trajectory)
