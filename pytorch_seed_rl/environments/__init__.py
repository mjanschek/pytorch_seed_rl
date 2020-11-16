"""The :py:mod:`~pytorch_seed_rl.environments` module includes all necessary functionality to spawn and wrap environments.

The module :py:mod:`~pytorch_seed_rl.environments.atari_wrappers` is a modified copy from the torchbeast project.

Exposed classes:
    * :py:class:`~EnvSpawner`

Unexposed modules:
    * :py:mod:`~pytorch_seed_rl.environments.atari_wrappers`

See Also
--------
`Github repository <https://github.com/facebookresearch/torchbeast>`__ of the torchbeast project.
"""

from .env_spawner import EnvSpawner
