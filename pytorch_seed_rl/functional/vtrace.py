# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Taken from
#   https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/vtrace.py
# and modified (mostly documentation)
"""Functions to compute V-trace off-policy actor critic targets.

See Also
--------
`"IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" 
on arXiv <https://arxiv.org/abs/1802.01561>`__ by Espeholt, Soyer, Munos et al.

All exposed functions return a :py:class:`~VTraceFromLogitsReturns`.
"""

import collections

import torch
import torch.nn.functional as F


def _action_log_probs(policy_logits: torch.Tensor,
                      actions: torch.Tensor) -> torch.Tensor:
    """Select the logits by their actions using negative log likelihood loss.
    """
    return -F.nll_loss(F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
                       torch.flatten(actions).to(torch.long),
                       reduction="none",
                       ).view_as(actions)


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)


def from_logits(behavior_policy_logits: torch.Tensor,
                target_policy_logits: torch.Tensor,
                values: torch.Tensor,
                bootstrap_value: torch.Tensor,
                actions: torch.Tensor,
                discounts: torch.Tensor,
                rewards: torch.Tensor,
                clip_rho_threshold: float = 1.0,
                clip_pg_rho_threshold: float = 1.0,
                ) -> VTraceFromLogitsReturns:
    """V-trace for softmax policies.

    Parameters
    ----------
    behavior_policy_logits: `torch.Tensor`
        The policies logits used for action sampling during interaction with the environment.
    target_policy_logits: `torch.Tensor`
        The policies logits returned by the learning model.
    values: `torch.Tensor`
        The values returned by the learning model.
    bootstrap_value: `torch.Tensor`
        The value used for bootstrapping (usually most recent value returned by learning model.)
    actions: `torch.Tensor`
        The actions used during interaction with the environment.
    discounts: `torch.Tensor`
        The discounted rewards.
    rewards: `torch.Tensor`
        The original rewards.
    clip_rho_threshold: `float`,
        Clipping value for Vtrace. See paper for details.
    clip_pg_rho_threshold: `float`,
        Clipping value for Vtrace. See paper for details.
    """
    # prepare logits
    target_action_log_probs = _action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = _action_log_probs(
        behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs

    vtrace_returns = _from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


_VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


@torch.no_grad()
def _from_importance_weights(log_rhos: torch.Tensor,
                             values: torch.Tensor,
                             bootstrap_value: torch.Tensor,
                             discounts: torch.Tensor,
                             rewards: torch.Tensor,
                             clip_rho_threshold: float = 1.0,
                             clip_pg_rho_threshold: float = 1.0,
                             ) -> _VTraceReturns:
    """V-trace from logarithmic importance weights.

    Parameters
    ----------
    log_rhos: `torch.Tensor`
        Logarithmic importance weights calculated from behaviour and target policy.
    values: `torch.Tensor`
        The values returned by the learning model.
    bootstrap_value: `torch.Tensor`
        The value used for bootstrapping (usually most recent value returned by learning model.)
    discounts: `torch.Tensor`
        The discounted rewards.
    rewards: `torch.Tensor`
        The original rewards.
    clip_rho_threshold: `float`,
        Clipping value for Vtrace. See paper for details.
    clip_pg_rho_threshold: `float`,
        Clipping value for Vtrace. See paper for details.
    """
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * \
            (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()

        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
        vs_t_plus_1 = torch.cat(
            [vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0
        )
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * \
            (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return _VTraceReturns(vs=vs, pg_advantages=pg_advantages)
