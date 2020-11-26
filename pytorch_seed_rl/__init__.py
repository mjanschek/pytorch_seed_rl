"""This package provides an extendable implementation of Google Brains "SEED".

It is designed to be capsuled to be extended in any way.
Good points of customization would be the neural netwok (:py:mod:`~pytorch_seed_rl.nets`)
or loss computation (:py:mod:`~pytorch_seed_rl.functional`).
Latter might need little modification of (:py:class:`~pytorch_seed_rl.agents.Learner`),
depending of the wanted change.

See Also
--------
* `"SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference"
  on arXiv <https://arxiv.org/abs/1910.06591>`__ by Espeholt et al. 
* `SEED Github repository <https://github.com/google-research/seed_rl>`__

Warnings
--------
Missing features in comparison to original implementation:
    * Algorithms `R2D2 <https://openreview.net/forum?id=r1lyTjAqYX>`__
      and `SAC <https://arxiv.org/abs/1801.01290>`__
    * LSTM model
    * Multi-Node distributed training
    * Rendering of actor-environment interaction
    * Tensorboard functionality
"""
