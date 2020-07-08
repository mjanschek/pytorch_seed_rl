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

"""Collection of pre-defined data structures.
"""

from typing import Any, NamedTuple, TypedDict, Union

from torch import Tensor

#from typing import List, TypedDict



class State(NamedTuple):
    """:py:class:`NamedTuple` to store a single state from an environment,
    AFTER inference has been run on it.

    Immutable type as intended use is storage within a :py:class:`Trajectory`
    """
    frame: Tensor
    reward: float
    done: bool
    episode_return: float
    episode_step: int
    last_action: Union[float, int]
    inference_info: Any
    metrics: Any


# class Trajectory(TypedDict):
#     """:py:class:`TypedDict` to store a number of observations of a single trajectory.

#     Mutable type as intended use is storage within a :py:class:`TrajectoryStore` while
#     :py:class:`Observation`s are appended.
#     """
#     global_trajectory_number: int
#     complete: bool
#     states: List[State]
