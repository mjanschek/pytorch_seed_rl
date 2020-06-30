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

"""The buffer class can be configured for various buffer types.
"""

class Buffer():
    """The buffer class is a generalization that can be configured to:
        #. be sampled from
        #. have a circular memory (consider IMPACT paper)
        #. have finite memory
    """

    def __init__(self):
        pass
