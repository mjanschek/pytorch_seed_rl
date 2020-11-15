"""The :py:mod:`environments` module includes all necessary functionality to spawn and wrap environments.

The module :py:mod:`~pytorch_seed_rl.environments.atari_wrappers` is a modified copy from the `torchbeast project <https://github.com/facebookresearch/torchbeast>`__.

Exposed classes:
    * :py:class:`~env_spawner.EnvSpawner`
"""

from .env_spawner import EnvSpawner
