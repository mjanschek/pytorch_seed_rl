"""Collection of neural networks intended for reinforcement learning.

Currently contains a single neural network architecture intended for usage with
:py:mod:`gym` environments.

Exposed Networks:
    * :py:class:`~atari_net.AtariNet`
"""

from .atari_net import AtariNet
