.. PyTorch SEED RL documentation master file, created by
   sphinx-quickstart on Thu Jul  2 15:25:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch SEED RL's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This package provides an extendable implementation of Google Brains "SEED".
The original implementation has been part of the master's thesis `Scaling reinforcement learning` by Michael Janschek.

It is designed to be capsuled to be extended in any way.
Good points of customization would be the neural netwok (:py:mod:`~pytorch_seed_rl.nets`) or loss computation (:py:mod:`~pytorch_seed_rl.functional`).
Latter might need little modification of (:py:class:`~pytorch_seed_rl.agents.Learner`), depending of the wanted change.

.. seealso::
   * The PyTorch SEED RL `Github repository <https://github.com/mjanschek/pytorch_seed_rl>`__
   * `"SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference" on arXiv <https://arxiv.org/abs/1910.06591>`__ by Espeholt et al. 
   * `SEED Github repository <https://github.com/google-research/seed_rl>`__

.. warning::
   Missing features in comparison to original implementation:
      * Algorithms `R2D2 <https://openreview.net/forum?id=r1lyTjAqYX>`__ and `SAC <https://arxiv.org/abs/1801.01290>`__
      * LSTM model
      * Multi-Node distributed training
      * Rendering of actor-environment interaction
      * Tensorboard functionality

Internal Links
==============
* :doc:`source/pytorch_seed_rl`
* :ref:`modindex`
* :ref:`Sitemap <genindex>`
