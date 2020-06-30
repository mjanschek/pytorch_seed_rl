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

"""The Model class acts as abstraction layer for all defined models in this module.
"""

from abc import abstractmethod

from torch import nn

class Model(nn.Module):
    """The Model class acts as abstraction layer for all defined models in this module.
    """

    # pylint: disable=invalid-name
    def forward(self, x):
        """Run inference on observation x.
        """
        return x

    @abstractmethod
    def _encoder(self):
        raise NotImplementedError

    @abstractmethod
    def _torso(self):
        raise NotImplementedError

    @abstractmethod
    def _policy_head(self):
        raise NotImplementedError

    @abstractmethod
    def _value_head(self):
        raise NotImplementedError
