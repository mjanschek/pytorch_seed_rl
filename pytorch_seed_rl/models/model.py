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
