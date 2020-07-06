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

from typing import Any, List, NamedTuple, TypedDict, Union

from torch import Tensor


class Observation(NamedTuple):
    """:py:class:`NamedTuple` to store a single observation from an environment,
    AFTER inference has been run on it.

    Immutable type as intended use is storage within a :py:class:`Trajectory`
    """

    step_number: int
    state: Tensor
    action: Union(float, int)
    reward: float
    inference_info: Any
    metrics: Any


class Trajectory(TypedDict):
    """:py:class:`TypedDict` to store a number of observations of a single trajectory.

    Mutable type as intended use is storage within a :py:class:`TrajectoryStore` while
    :py:class:`Observation`s are appended.
    """

    agent_id: int
    agent_env_id: int = 0
    complete: bool
    global_trajectory_number: int
    observations: List[Observation]
