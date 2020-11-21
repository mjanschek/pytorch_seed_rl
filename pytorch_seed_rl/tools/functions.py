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
#
# Partly taken from https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists

"""Collection of minor utility/qol functions
"""

import functools
from typing import Dict, List


def listdict_to_dictlist(listdict: List[dict]) -> Dict[str, list]:
    """Transforms a list of dictionaries into a dictionary of lists.

    Parameters
    ----------
    listdict: 'list' of 'dict'
        A list of dictionaries
    """
    return {key: [item[key] for item in listdict]
            for key in list(functools.reduce(lambda x, y: x.union(y),
                                             (set(dicts.keys())
                                              for dicts in listdict)
                                             ))
          }

    # return {k: [dic[k] for dic in listdict] for k in listdict[0]}


def dictlist_to_listdict(dictlist: Dict[str, list]) -> List[dict]:
    """Transforms a dictionary of lists into a list of dictionaries.

    Parameters
    ----------
    listdict: 'dict' of 'list'
        A dictionary of lists
    """
    return [dict(zip(dictlist, t)) for t in zip(*dictlist.values())]
