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
"""Collection of loss functions necessary for reinforcement learning objective calculations.
"""

import torch
import torch.nn.functional as F


# def gae_v():
#     """Consider GAE_V paper.
#     """


# def compute_baseline_loss(advantages):
#     return 0.5 * torch.sum(advantages ** 2)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return the entropy loss, i.e., the negative entropy of the policy.

    This can be used to discourage an RL model to converge prematurely.

    See Also
    --------
    `Entropy Regularization in Reinforcement Learning <https://towardsdatascience.com/entropy-regularization-in-reinforcement-learning-a6fa6d7598df>`__

    Parameters
    ----------
    logits: :py:class:`torch.Tensor`
        Logits returned by the models policy network.
    """
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def policy_gradient(logits: torch.Tensor,
                    actions: torch.Tensor,
                    advantages: torch.Tensor) -> torch.Tensor:
    """Compute the policy gradient loss.

    See Also
    --------
    `https://spinningup.openai.com <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>`__

    Parameters
    ----------
    logits: :py:class:`torch.Tensor`
        Logits returned by the models policy network.
    actions: :py:class:`torch.Tensor`
        Actions that were selected from :py:attr:`logits`
    advantages: :py:class:`torch.Tensor`
        Advantages that resulted for the related states.
    """
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1).to(torch.long),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())
