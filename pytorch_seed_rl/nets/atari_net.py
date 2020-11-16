# Copyright (c) Facebook, Inc. and its affiliates.
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

# taken from https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/monobeast.py
#   and modified (mostly documentation)
"""test
"""
import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F


class AtariNet(Module):
    """Neural network architecture intended for usage with :py:mod:`gym` environments.

    This network architecture is copied from the torchbeast project,
    which mimics the neural network used in the IMPALA paper.

    See Also
    --------
    * `"IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" <https://arxiv.org/abs/1802.01561>`__ by Espeholt, Soyer, Munos et al. 
    * `Torchbeast implementation <https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/monobeast.py>`__

    Parameters
    ----------
    observation_shape: `tuple`
        The shape of the tensors this neural network processes.
    num_actions: `int`
        The number of discrete actions this neural network can return.
    use_lstm: `bool`
        Set True, if an LSTM shall be included with this neural network.
    """

    def __init__(self,
                 observation_shape: tuple,
                 num_actions: int,
                 use_lstm: bool = False):
        super(AtariNet, self).__init__()
        # ATTRIBUTES
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size: int):
        """Return 0 :py:obj:`torch.Tensor` with shape of LSTM block.

        Returns None, if LSTM has not been activated during initialization.
        """
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size,
                        self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs: dict, core_state: tuple = ()):
        """Forward step of the neural network

        Parameters
        ----------
        inputs: `dict` of :py:obj:`torch.Tensor`
            Awaits a dictionary as returned by an step of :py:class:`~pytorch_seed_rl.environments.atari_wrappers.DictObservationsEnv`
        """
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.leaky_relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat(
            [x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for c_input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(c_input.unsqueeze(0),
                                               core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            probs = F.softmax(policy_logits, dim=1)
            action = torch.multinomial(probs, num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )
