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

Usage
=====

Run Experiments
---------------
You can launch a default experiment from the CLI using:

>>> python -m pytorch_seed_rl.run ExperimentName

This will create a saving directory in ``~/logs/pytorch_seed_rl/ExperimentName``, where all data generated will be saved.

Full printout of the functions help:

>>> python -m pytorch_seed_rl.run -h

.. code-block:: python

   PyTorch_SEED_RL

   positional arguments:
   name                  Experiments name, defaults to environment id.

   optional arguments:
   -h, --help            show this help message and exit
   -R, --reset           USE WITH CAUTION! Resets existing experiment, this
                           removes all data on subdir level.
   -C, --load_checkpoint
                           Continues training, if model checkpoint is available.
   -v, --verbose         Prints system metrics to command line.Set
                           --print_interval for number of training epochs between
                           prints.
   --print_interval PRINT_INTERVAL
                           Number of training epochs between prints.
   --system_log_interval SYSTEM_LOG_INTERVAL
                           Number of core loops between logging system metrics.
   --checkpoint_interval CHECKPOINT_INTERVAL
                           Number of training epochs between checkpointing.
   --savedir SAVEDIR     Root dir where experiment data will be saved.
   --render              Renders an episode as gif, if the recorded data
                           finished with a new point record.
   --max_gif_length MAX_GIF_LENGTH
                           Enforces a maximum gif length.Rendering is triggered,
                           if recorded data reaches this volume.
   --gpu_ids GPU_IDS     A comma-separated list of cuda ids this program is
                           permitted to use.
   --total_steps TOTAL_STEPS
                           Total environment steps.
   --max_epoch MAX_EPOCH
                           Training epoch limit. Set to -1 for no limit.
   --max_time MAX_TIME   Runtime limit. Set to -1 for no limit.
   --batchsize_training BATCHSIZE_TRAINING
                           Training batch size.
   --rollout ROLLOUT     The rollout length used for training. See IMPALA paper
                           for more info.
   --use_lstm            Use LSTM in agent model.
   --env ENV             Gym environment.
   --num_envs NUM_ENVS   Number of environments per actor.
   --master_address MASTER_ADDRESS
                           The master adress for the RPC processgroup. WARNING:
                           CHANGE WITH CAUTION!
   --master_port MASTER_PORT
                           The master port for the RPC processgroup. WARNING:
                           CHANGE WITH CAUTION!
   --num_actors NUM_ACTORS
                           Number of actors.
   --threads_prefetch THREADS_PREFETCH
                           Number of prefetch threads.
   --threads_inference THREADS_INFERENCE
                           Number of inference threads.
   --threads_store THREADS_STORE
                           Number of storing threads.
   --tensorpipe          Uses the default RPC backend of pytorch, Tensorpipe.
   --max_queued_batches MAX_QUEUED_BATCHES
                           Number of batches that can be queued concurrently.This
                           prevents memory overflow.
   --max_queued_drops MAX_QUEUED_DROPS
                           Number of trajectories that can be queued concurrently
                           by the store.This prevents memory overflow.
   --pg_cost PG_COST     Policy gradient cost/multiplier.
   --baseline_cost BASELINE_COST
                           Baseline cost/multiplier.
   --entropy_cost ENTROPY_COST
                           Entropy cost/multiplier.
   --discounting DISCOUNTING
                           Discounting factor.
   --reward_clipping {abs_one,none}
                           Reward clipping.
   --optimizer {adam,rmsprop}
                           Optimizer used for weight updates.
   --learning_rate LR    Learning rate.
   --grad_norm_clipping GRAD_NORM_CLIPPING
                           Global gradient norm clip.
   --epsilon EPSILON     Optimizer epsilon for numerical stability.
   --decay DECAY         Optimizer weight decay.
   --beta_1 BETA_1       Adam beta 1.
   --beta_2 BETA_2       Adam beta 2.
   --alpha ALPHA         RMSProp smoothing constant.
   --momentum MOMENTUM   RMSProp momentum.


Evaluate Models
---------------
You can evaluate a model with the ``eval`` module from the CLI using:

>>> python -m pytorch_seed_rl.eval ExperimentName

This will read the model at ``~/logs/pytorch_seed_rl/ExperimentName/model/final_model.pt``.

Full printout of the functions help:

>>> python -m pytorch_seed_rl.eval -h

.. code-block:: python

   PyTorch_SEED_RL

   positional arguments:
   name                  Experiments name, defaults to environment id.

   optional arguments:
   -h, --help            show this help message and exit
   -v, --verbose         Prints system metrics to command line.Set
                           --print_interval for number of training epochs between
                           prints.
   --total_steps TOTAL_STEPS
                           Total environment steps.
   --env ENV             Gym environment.
   --print_interval PRINT_INTERVAL
                           Number of training epochs between prints.
   --savedir SAVEDIR     Root dir where experiment data will be saved.
   --render              Renders an episode as gif, if the recorded data
                           finished with a new point record.
   --max_gif_length MAX_GIF_LENGTH
                           Enforces a maximum gif length.Rendering is triggered,
                           if recorded data reaches this volume.
   --gpu_ids GPU_IDS     A comma-separated list of cuda ids this program is
                           permitted to use.
