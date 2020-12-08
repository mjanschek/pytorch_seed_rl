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
The original implementation has been part of the master's thesis `Scaling Reinforcement Learning` by Michael Janschek.

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
      * Tensorboard functionality
                        
Internal Links
==============
* :doc:`source/pytorch_seed_rl`
* :ref:`modindex`
* :ref:`Sitemap <genindex>`

Quickstart Guide
================
To install this tool, clone the Git repository from Github and install directly from the source using pip:

.. code-block::

   > git clone https://github.com/mjanschek/pytorch_seed_rl.git
   > cd pytorch_seed_rl
   > pip install .

Running an Experiment
---------------------
You can launch a default experiment from the CLI using:

.. code-block::

   > python -m pytorch_seed_rl.run ExperimentName

This will create a saving directory at the path ``~/logs/pytorch_seed_rl/ExperimentName``, where all data generated will be saved.
This defaults to being log data that is written to csv files in the subdirectory ``/csv/``. If the flag ``--render`` is used,
the algorithm will create gif files of episodes that achieved a new record return.
These gifs are saved in another subdirectory ``/gif/``.
Note that frames that are used for gifs are copied from the inference pipeline,
this implies that all preprocessing of environment states also affect the frames used for a gif.

For information about the available CLI arguments, we refer the user to the help flag:

.. code-block::

   > python -m pytorch_seed_rl.run -h


Evaluate Models
---------------
You can evaluate a model with the ``eval`` module from the CLI using:

>>> python -m pytorch_seed_rl.eval ExperimentName


This will search for the file saving directory at the path
``~/logs/pytorch_seed_rl/ExperimentName/model/final_model.pt``
that is always created after an experiment conducted with this project reached
one of its shutdown criteria.

If a model file is found, the function will run a simple interaction loop using
a single actor and a single environment.
A subdirectory ``/eval/`` is created within the experiments folder.
There, the subdirectories ``/csv/`` and ``/gif/`` are created as needed,
depending on set flags. Note that the ``--render`` flag
does record **every** episode the actor plays.
Frames that are used for gifs are copied from the inference pipeline,
this implies that all preprocessing of environment states also affect
the frames used for a gif.

For information about the available CLI arguments, we refer the user to the help flag:

.. code-block::

   > python -m pytorch_seed_rl.eval -h