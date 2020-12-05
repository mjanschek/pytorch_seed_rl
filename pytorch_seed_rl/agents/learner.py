# Copyright 2020 Michael Janschek
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
#
# pylint: disable=empty-docstring
"""
"""
import copy
import gc
import os
import pprint
import queue
import time
import warnings
from collections import deque
from threading import Thread
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR

from .. import agents
from ..agents.rpc_callee import RpcCallee
from ..environments import EnvSpawner
from ..functional import loss, vtrace
from ..tools import Recorder, TrajectoryStore
from ..tools.functions import listdict_to_dictlist


class Learner(RpcCallee):
    """Agent that runs inference and learning in parallel.

    This learning agent implements the reinforcement learning algorithm IMPALA
    following the SEED RL implementation by Google Brain.

    During initiation:
        * Spawns :py:attr:`num_actors` instances of :py:class:`~.agents.Actor`.
        * Invokes their :py:meth:`~.RpcCaller.loop()` methods.
        * Creates a :py:class:`~.Recorder`.
        * Creates a :py:class:`~.TrajectoryStore`.
        * Starts a continous inference process to answer pending RPCs.
        * Starts a continous prefetching process to prepare batches
          of complete trajectories for learning.

    During runtime:
        * Runs evaluates observations received from
          :py:class:`~.agents.Actor` and returns actions.
        * Stores incomplete trajectories in :py:class:`~.TrajectoryStore`.
        * Trains a global model from trajectories received from a data prefetching thread.

    Parameters
    ----------
    rank : `int`
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`).
    num_actors : `int`
        Number of total :py:class:`~.agents.Actor` objects to spawn.
    env_spawner : :py:class:`~.EnvSpawner`
        Object that spawns an environment on invoking it's
        :py:meth:`~.EnvSpawner.spawn()` method.
    model : :py:class:`torch.nn.Module`
        A torch model that processes frames as returned
        by an environment spawned by :py:attr:`env_spawner`
    optimizer : :py:class:`torch.nn.Module`
        A torch optimizer that links to :py:attr:`model`.
    save_path : `str`
        The root directory for saving data. Default: the current working directory.
    pg_cost : `float`
        Policy gradient cost/multiplier.
    baseline_cost : `float`
        Baseline cost/multiplier.
    entropy_cost : `float`
        Entropy cost/multiplier.
    grad_norm_clipping : `float`
        If bigger 0, clips the computed gradient norm to given maximum value.
    reward_clipping : `bool`
        Reward clipping.
    batchsize_training : `int`
        Number of complete trajectories to gather before learning from them as batch.
    rollout : `int`
        Length of rollout used by the IMPALA algorithm.
    total_steps : `int`
        Maximum number of environment steps to learn from.
    max_epoch : `int`
        Maximum number of training epochs to do.
    max_time : `int`
        Maximum time for training.
    threads_prefetch : `int`
        The number of threads that shall prefetch data for training.
    threads_inference : `int`
        The number of threads that shall perform inference.
    threads_store : `int`
        The number of threads that shall store data into trajectory store.
    render: `bool`
        Set True, if episodes shall be rendered.
    max_gif_length: `bool`
        The maximum number of frames that shall be saved as a single gif.
        Set to 0 (default), if no limit shall be enforced.
    verbose : `bool`
        Set True if system metrics shall be printed at interval set by `print_interval`.
    print_interval : `int`
        Interval of training epoch system metrics shall be printed. Set to 0 to surpress printing.
    system_log_interval : `int`
        Interval of logging system metrics.
    load_checkpoint : `bool`
        Set True if the most checkpoint shall be loaded.
    checkpoint_interval : `int`
        Interval of checkpointing. Set to 0 to surpress checkpointing.
    max_queued_batches: `int`
        Limits the number of batches that can be queued at once.
    max_queued_drops: `int`
        Limits the number of dropped trajectories that can be queued by the trajectory store.
    max_queued_stores: `int`
        Limits the number of states that can be queued to be stored.
    """

    def __init__(self,
                 rank: int,
                 num_actors: int,
                 env_spawner: EnvSpawner,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_path: str = ".",
                 pg_cost: float = 1.,
                 baseline_cost: float = 1.,
                 entropy_cost: float = 0.01,
                 discounting: float = 0.99,
                 grad_norm_clipping: float = 40.,
                 reward_clipping: bool = True,
                 batchsize_training: int = 4,
                 rollout: int = 80,
                 total_steps: int = -1,
                 max_epoch: int = -1,
                 max_time: float = -1.,
                 threads_prefetch: int = 1,
                 threads_inference: int = 1,
                 threads_store: int = 1,
                 render: bool = False,
                 max_gif_length: int = 0,
                 verbose: bool = False,
                 print_interval: int = 10,
                 system_log_interval: int = 1,
                 load_checkpoint: bool = False,
                 checkpoint_interval: int = 10,
                 max_queued_batches: int = 128,
                 max_queued_drops: int = 128,
                 max_queued_stores: int = 1024):

        self.total_num_envs = num_actors*env_spawner.num_envs
        self.envs_list = [i for i in range(self.total_num_envs)]

        super().__init__(rank,
                         num_callees=1,
                         num_callers=num_actors,
                         threads_process=threads_inference,
                         caller_class=agents.Actor,
                         caller_args=[env_spawner],
                         future_keys=self.envs_list)

        # ATTRIBUTES
        self._save_path = save_path
        self._model_path = os.path.join(save_path, 'model')

        self._pg_cost = pg_cost
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost
        self._discounting = discounting
        self._grad_norm_clipping = grad_norm_clipping
        self._reward_clipping = reward_clipping
        self._batchsize_training = batchsize_training
        self._rollout = rollout

        self._total_steps = total_steps
        self._max_epoch = max_epoch
        self._max_time = max_time

        self._verbose = verbose
        self._print_interval = print_interval
        self._system_log_interval = system_log_interval
        self._checkpoint_interval = checkpoint_interval

        # COUNTERS
        self.inference_epoch = 0
        self.inference_steps = 0
        self.inference_time = 0.

        self.training_epoch = 0
        self.training_steps = 0
        self.training_time = 0.

        self.fetching_time = 0.

        self.runtime = 0

        self.dead_counter = 0

        # TORCH
        self.training_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # use 2 gpus if available
        if torch.cuda.device_count() < 2:
            self.eval_device = self.training_device
        else:
            print("2 GPUs used!")
            self.eval_device = torch.device("cuda:1")
            model = DataParallel(model)

        self.model = model.to(self.training_device)

        self.eval_model = copy.deepcopy(self.model)
        self.eval_model = self.eval_model.to(self.eval_device)
        self.eval_model.eval()

        self.optimizer = optimizer

        # define a linear decreasing function for linear scheduler
        def linear_lambda(epoch):
            return 1 - min(epoch * rollout * batchsize_training, total_steps) / total_steps
        self.scheduler = LambdaLR(self.optimizer, linear_lambda)

        # TOOLS
        self.recorder = Recorder(save_path=self._save_path,
                                 render=render,
                                 max_gif_length=max_gif_length)

        # LOAD CHECKPOINT, IF WANTED
        if load_checkpoint:
            self._load_checkpoint(self._model_path)

        # THREADS
        self.lock_model = mp.Lock()
        self.lock_prefetch = mp.Lock()
        self.shutdown_event = mp.Event()

        self.queue_drops = mp.Queue(maxsize=max_queued_drops)
        self.queue_batches = mp.Queue(maxsize=max_queued_batches)
        self.storing_deque = deque(maxlen=max_queued_stores)

        # check variables used by _check_dead_queues()
        self.queue_batches_old = self.queue_batches.qsize()
        self.queue_drop_off_old = self.queue_drops.qsize()
        self.queue_rpcs_old = len(self._pending_rpcs)

        # Create prefetch threads
        # Not actual threads. Name is chosen due to equal API usage in this code
        self.prefetch_threads = [mp.Process(target=self._prefetch,
                                            args=(self.queue_drops,
                                                  self.queue_batches,
                                                  batchsize_training,
                                                  self.shutdown_event,
                                                  self.training_device),
                                            daemon=True,
                                            name='prefetch_thread_%d' % i)
                                 for i in range(threads_prefetch)]

        self.storing_threads = [Thread(target=self._store,
                                       daemon=True,
                                       name='storing_thread_%d' % i)
                                for i in range(threads_store)]

        # spawn trajectory store
        placeholder_eval_obs = self._build_placeholder_eval_obs(env_spawner)
        self.trajectory_store = TrajectoryStore(self.envs_list,
                                                placeholder_eval_obs,
                                                self.eval_device,
                                                self.queue_drops,
                                                self.recorder,
                                                trajectory_length=rollout)

        # start actors
        self._start_callers()

        # start threads and processes
        for thread in [*self.prefetch_threads, *self.storing_threads]:
            thread.start()

    # pylint: disable=arguments-differ
    def _loop(self, waiting_time: float = 5):
        """Inner loop function of a :py:class:`Learner`.

        Called by :py:meth:`.RpcCallee.loop()`.

        This method first pulls a batch in :py:attr:`self.queue_batches`.
        Then it invokes :py:meth:`_learn_from_batch()`
        and copies the updated model weights from the learning model to :py:attr:`self.eval_model`.
        System metrics are passed logged using :py:meth:`~.Recorder.log()`.
        Finally, it checks for reached shutdown criteria,
        like :py:attr:`self._total_steps` has been reached.

        Parameters
        ----------
        waiting_time: `float`
            Seconds to wait on batches delivered by :py:attr:`self.queue_batches`.
        """
        batch = None
        try:
            batch = self.queue_batches.get(timeout=waiting_time)
        except queue.Empty:
            pass

        if batch is not None:
            training_metrics = self._learn_from_batch(batch,
                                                      grad_norm_clipping=self._grad_norm_clipping,
                                                      pg_cost=self._pg_cost,
                                                      baseline_cost=self._baseline_cost,
                                                      entropy_cost=self._entropy_cost)

            # delete Tensors after usage to free memory (see torch multiprocessing)
            del batch

            with self.lock_model:
                self.eval_model.load_state_dict(self.model.state_dict())

            self.recorder.log('training', training_metrics)

        if self._checkpoint_interval > 0:
            # mus split up to not divide by zero
            if self.training_epoch % self._checkpoint_interval == 0:
                # 0 or 1
                i = int(self.training_epoch %
                        (2*self._checkpoint_interval) != 0)
                self._save_checkpoint(self._model_path, i)

        # check if queues are dead
        self._check_dead_queues()

        # check, if shutdown prerequisites haven been reached
        self.shutdown = ((self.training_epoch > self._max_epoch > 0) or
                         (self.training_steps > self._total_steps > 0) or
                         (self.get_runtime() > self._max_time > 0) or
                         self.shutdown)

        if self._loop_iteration == self._system_log_interval and not self.shutdown:
            system_metrics = self._get_system_metrics()
            self.recorder.log('system', system_metrics)
            self._loop_iteration = 0

            if self._verbose and (self.training_epoch % self._print_interval == 0):
                print(pprint.pformat(system_metrics))

        # be sure to broadcast or react to shutdown event
        if self.shutdown_event.is_set():
            self.shutdown = True
        elif self.shutdown:
            self.shutdown_event.set()

    def process_batch(self,
                      caller_ids: List[Union[int, str]],
                      *batch: List[dict],
                      **misc: dict) -> Dict[str, torch.Tensor]:
        """Inner method to process a whole batch at once.

        Called by :py:meth:`~.RpcCallee._process_batch()`.

        Before returning the result for the given batch, this method:
            # . Moves its data to the :py:class:`Learner` device (usually GPU)
            # . Runs inference on this data
            # . Invokes :py:meth:`_queue_for_storing()` to put evaluated data on
               storing queue.

        Parameters
        ----------
        caller_ids : `list[int]` or `list[str]`
            List of unique identifiers for callers.
        batch : `list[dict]`
            List of inputs for evaluation.
        misc : `dict`
            Dict of keyword arguments. Primarily used for metrics in this application.
        """
        # concat tensors for each dict in a batch and move to own device
        for dictionary in batch:
            for key, value in dictionary.items():
                try:
                    # [T, B, C, H, W] => [1, batchsize, C, H, W]
                    dictionary[key] = torch.cat(
                        value, dim=1).to(self.eval_device)
                except TypeError:
                    # expected for input dictionaries that are not tensors
                    continue

        # more arguments could be sotred in batch tuple
        states = batch[0]

        # run inference
        start = time.time()
        with self.lock_model:
            inference_output, _ = self.eval_model(states)
        self.inference_time += time.time() - start

        # log model state at time of inference
        inference_output['training_steps'] = torch.zeros_like(
            states['episode_return']).fill_(self.training_steps)

        self.inference_steps += states['frame'].shape[1]
        self.inference_epoch += 1

        # add states to store in parallel process. Don't move data via RPC as it shall stay on cuda.
        states = {k: v.detach()
                  for k, v in {**states, **inference_output}.items()}

        metrics = misc['metrics']

        for i, i_caller_id in enumerate(caller_ids):
            self._queue_for_storing(i_caller_id,
                                    {k: v[0, i] for k, v in states.items()},
                                    metrics[i])

        # gather an return results
        results = {c: inference_output['action'][0][i].view(
            1, 1).cpu().detach() for i, c in enumerate(caller_ids)}

        return results

    def _queue_for_storing(self,
                           caller_id: str,
                           state: dict,
                           metrics: dict):
        """Wrap for storing data onto the :py:obj:`~self.storing_deque`.

        Parameters
        ----------
        caller_id: `int` or `str`
            The caller id of the environments the data belongs to.
            Necessary for proper storing.
        state: `dict`
            An environments state dictionary.
        metrics: `dict`
            An actors metrics dictionary.
        """
        while True and not self.shutdown:
            if len(self.storing_deque) < self.storing_deque.maxlen:
                self.storing_deque.append((caller_id, state, metrics))
                break
            else:
                time.sleep(0.001)

    def _store(self, waiting_time: float = 0.1):
        """Periodically checks for data in :py:obj:`self.storing_deque`
        and stores found data into the :py:class:`~.TrajectoryStore`.

        Intended for use as :py:obj:`multiprocessing.Process`.

        Parameters
        ----------
        waiting_time: `float`
            The time in seconds to wait for new data.
        """
        while not self.shutdown_event.is_set():
            try:
                caller_id, state, metrics = self.storing_deque.popleft()
            except IndexError:  # deque empty
                time.sleep(waiting_time)
                continue
            self.trajectory_store.add_to_entry(caller_id, state, metrics)
            del state, metrics

    def _learn_from_batch(self,
                          batch: Dict[str, torch.Tensor],
                          grad_norm_clipping: float = 40.,
                          pg_cost: float = 1.,
                          baseline_cost: float = 0.5,
                          entropy_cost: float = 0.01) -> Dict[str, Any]:
        """Runs the learning process and updates the internal model.

        This method:
            # . Evaluates the given :py:attr:`batch` with the internal learning model.
            # . Invokes :py:meth:`compute_losses()` to get all components of the loss function.
            # . Calculates the total loss, using the given cost factors for each component.
            # . Updates the model by invoking the :py:attr:`self.optimizer`.

        Parameters
        ----------
        batch : `dict`
            Dict of stacked tensors of complete trajectories as returned by :py:meth:`_to_batch()`.
        grad_norm_clipping : `float`
            If bigger 0, clips the computed gradient norm to given maximum value.
        pg_cost : `float`
            Cost/Multiplier for policy gradient loss.
        baseline_cost : `float`
            Cost/Multiplier for baseline loss.
        entropy_cost : `float`
            Cost/Multiplier for entropy regularization.
        """
        # evaluate training batch
        batch_length = batch['current_length'].sum().item()
        learner_outputs, _ = self.model(batch)

        pg_loss, baseline_loss, entropy_loss = self.compute_losses(
            batch,
            learner_outputs,
            discounting=self._discounting,
            reward_clipping=self._reward_clipping
        )

        total_loss = pg_cost * pg_loss \
            + baseline_cost * baseline_loss \
            + entropy_cost * entropy_loss

        self.training_steps += batch_length
        self.training_epoch += 1

        # perform update
        self.optimizer.zero_grad()
        total_loss.backward()
        if grad_norm_clipping > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), grad_norm_clipping)
        self.optimizer.step()
        self.scheduler.step()

        return {"runtime": self.get_runtime(),
                "training_time": self.training_time,
                "training_epoch": self.training_epoch,
                "training_steps": self.training_steps,
                "total_loss": total_loss.detach().cpu().item(),
                "pg_loss": pg_loss.detach().cpu().item(),
                "baseline_loss": baseline_loss.detach().cpu().item(),
                "entropy_loss": entropy_loss.detach().cpu().item(),
                }

    @staticmethod
    def compute_losses(batch: Dict[str, torch.Tensor],
                       learner_outputs: Dict[str, torch.Tensor],
                       discounting: float = 0.99,
                       reward_clipping: bool = True
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes and returns the components of IMPALA loss.

        Calculates policy gradient, baseline and entropy loss using Vtrace for value estimation.

        See Also
        --------
        * The :py:mod:`.functional.loss` module.
        * The :py:mod:`.functional.vtrace` module.

        Parameters
        ----------
        batch : `dict`
            Dict of stacked tensors of complete trajectories as returned by :py:meth:`_to_batch()`.
        learner_outputs : `dict`
            Dict with outputs generated during evaluation within training.
        discounting : `float`
            Reward discout factor, must be a positive smaller than 1.
        reward_clipping : `bool`
            If set, rewards are clamped between -1 and 1.
        """
        assert 0 < discounting <= 1.

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1]
                           for key, tensor in learner_outputs.items()}

        # clip rewards, if wanted
        if reward_clipping:
            batch["reward"] = torch.clamp(batch["reward"], -1, 1)

        discounts = (~batch["done"]).float() * discounting

        vtrace_returns = vtrace.from_logits(behavior_policy_logits=batch["policy_logits"],
                                            target_policy_logits=learner_outputs["policy_logits"],
                                            values=learner_outputs["baseline"],
                                            bootstrap_value=bootstrap_value,
                                            actions=batch["action"],
                                            rewards=batch["reward"],
                                            discounts=discounts,)

        pg_loss = loss.policy_gradient(learner_outputs["policy_logits"],
                                       batch["action"],
                                       vtrace_returns.pg_advantages)

        baseline_loss = F.mse_loss(learner_outputs["baseline"],
                                   vtrace_returns.vs,
                                   reduction='sum')

        entropy_loss = loss.entropy(learner_outputs["policy_logits"])

        return pg_loss, baseline_loss, entropy_loss

    @staticmethod
    def _prefetch(in_queue: mp.Queue,
                  out_queue: mp.Queue,
                  batchsize: int,
                  shutdown_event: mp.Event,
                  target_device,
                  waiting_time=5):
        """Continuously prefetches complete trajectories dropped by
        the :py:class:`~.TrajectoryStore` for training.

        As long as shutdown is not set, this method
        pulls :py:attr:`batchsize` trajectories from :py:attr:`in_queue`,
        transforms them into batches using :py:meth:`~_to_batch()`
        and puts them onto the :py:attr:`out_queue`.

        This usually runs as an asynchronous :py:obj:`multiprocessing.Process`.

        Parameters
        ----------
        in_queue: :py:obj:`multiprocessing.Queue`
            A queue that delivers dropped trajectories from :py:class:`~.TrajectoryStore`.
        out_queue: :py:obj:`multiprocessing.Queue`
            A queue that delivers batches to :py:meth:`_loop()`.
        batchsize: `int`
            The number of trajectories that shall be processed into a batch.
        shutdown_event: :py:obj:`multiprocessing.Event`
            An event that breaks this methods internal loop.
        target_device: :py:obj:`torch.device`
            The target device of the batch.
        waiting_time: `float`
            Time the methods loop sleeps between each iteration.
        """

        while not shutdown_event.is_set():
            try:
                trajectories = [in_queue.get(timeout=waiting_time)
                                for _ in range(batchsize)]
            except queue.Empty:
                continue

            batch = Learner._to_batch(trajectories, target_device)
            # delete Tensors after usage to free memory (see torch multiprocessing)
            del trajectories

            try:
                out_queue.put(batch)
            except (AssertionError, ValueError):  # queue closed
                continue

        # delete Tensors after usage to free memory (see torch multiprocessing)
        del batch
        try:
            del trajectories
        except UnboundLocalError:  # already deleted
            pass

    @staticmethod
    def _to_batch(trajectories: List[dict], target_device) -> Dict[str, torch.Tensor]:
        """Extracts states from a list of trajectories, returns them as batch.

        Parameters
        ----------
        trajectories: `list`
            List of trajectories dropped by :py:class:`~.TrajectoryStore`.
        target_device: :py:obj:`torch.device`
            The target device of the batch.
        """
        states = listdict_to_dictlist([t['states'] for t in trajectories])

        for key, value in states.items():
            # [T, B, C, H, W]  => [len(trajectories), batchsize, C, H, W]
            states[key] = torch.cat(value, dim=1).clone().to(target_device)

        states['current_length'] = torch.stack(
            [t['current_length'] for t in trajectories]).clone()

        return states

    def _get_system_metrics(self):
        """Returns the training systems metrics.
        """
        return {
            "runtime": self.get_runtime(),
            "trajectories_seen": self.recorder.trajectories_seen,
            "episodes_seen": self.recorder.episodes_seen,
            "mean_inference_latency": self.recorder.mean_latency,
            "fetching_time": self.fetching_time,
            "inference_time": self.inference_time,
            "inference_steps": self.inference_steps,
            "training_time": self.training_time,
            "training_steps": self.training_steps,
            "queue_batches": self.queue_batches.qsize(),
            "queue_drops": self.queue_drops.qsize(),
            "queue_rpcs": len(self._pending_rpcs),
            "queue_storing": len(self.storing_deque),
        }

    def _save_model(self,
                    path: str,
                    filename: str = 'final_model.pt'):
        """Save the model at the given path.

        Parameters
        ----------
        path: `str`
            A valid path to a directory.
        filename: `str`
            The filename the saved model shall have. Defaults to ``final_model.pt``.
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        torch.save(self.model.state_dict(), save_path)

        print('Final model saved at %s.' % save_path)

    def _save_checkpoint(self,
                         path: str,
                         i: int):
        """Saves a checkpoint at the given path.

        The filename is fixed to ``checkpoint_i.pt``. ``i`` being an integer.

        Parameters
        ----------
        path: `str`
            A valid path to a directory.
        i: `str`
            The filenames suffix. This is used to enable multiple consecutive checkpoints.
        """
        with warnings.catch_warnings():
            # warning thrown on scheduler.state_dict(): optimizers state should be saved as well.
            # disable this warning because we do save the optimizers state
            warnings.simplefilter("ignore", category=UserWarning)
            model_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }

        training_dict = {
            "_pg_cost": self._pg_cost,
            "_baseline_cost": self._baseline_cost,
            "_entropy_cost": self._entropy_cost,
            "_discounting": self._discounting,
            "_grad_norm_clipping": self._grad_norm_clipping,
            "_reward_clipping": self._reward_clipping,
            "_batchsize_training": self._batchsize_training,
            "_rollout": self._rollout,
            "_total_steps": self._total_steps,
        }

        counters_dict = {
            "runtime": self.get_runtime(),
            "trajectories_seen": self.recorder.trajectories_seen,
            "episodes_seen": self.recorder.episodes_seen,
            "training_steps": self.training_steps,
            "training_epoch": self.training_epoch,
            "inference_steps": self.inference_steps,
        }

        save_dict = {
            **model_dict,
            **training_dict,
            **counters_dict
        }

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, 'checkpoint_%d.pt' % i)

        torch.save(save_dict, save_path)

        if self._verbose:
            print('Made checkpoint after training epoch %d.' %
                  self.training_epoch)

    def _load_checkpoint(self,
                         path: str):
        """Loads the latest checkpoint from the given path.
        Data will be loaded directly into programs components.

        The files 'checkpoint_0.pt' and 'checkpoint_1.pt' will be checked at the given path.

        Parameters
        ----------
        path: `str`
            A valid path to a directory.
        """
        try:
            checkpoint_0 = torch.load(os.path.join(path, 'checkpoint_0.pt'))
        except FileNotFoundError:
            checkpoint_0 = None
        try:
            checkpoint_1 = torch.load(os.path.join(path, 'checkpoint_1.pt'))
        except FileNotFoundError:
            checkpoint_1 = None

        if (checkpoint_0 is not None
                and (checkpoint_1 is None
                     or checkpoint_0['training_epoch'] >= checkpoint_1['training_epoch'])):
            checkpoint = checkpoint_0
            del checkpoint_1
        elif (checkpoint_1 is not None
              and (checkpoint_0 is None
                   or checkpoint_1['training_epoch'] > checkpoint_0['training_epoch'])):
            checkpoint = checkpoint_1
            del checkpoint_0
        else:
            raise FileNotFoundError('No checkpoints found!')

            # model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.eval_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with warnings.catch_warnings():
            # warning thrown on scheduler.state_dict(): optimizers state should be loaded as well.
            # disable this warning because we do save the optimizers state
            warnings.simplefilter("ignore", category=UserWarning)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # training
        self._pg_cost = checkpoint['_pg_cost']
        self._baseline_cost = checkpoint['_baseline_cost']
        self._entropy_cost = checkpoint['_entropy_cost']
        self._discounting = checkpoint['_discounting']
        self._grad_norm_clipping = checkpoint['_grad_norm_clipping']
        self._reward_clipping = checkpoint['_reward_clipping']
        self._batchsize_training = checkpoint['_batchsize_training']
        self._rollout = checkpoint['_rollout']

        # counters
        self._t_start = time.time()-checkpoint['runtime']
        self.recorder.trajectories_seen = checkpoint['trajectories_seen']
        self.recorder.episodes_seen = checkpoint['episodes_seen']
        self.training_steps = checkpoint['training_steps']
        self.training_epoch = checkpoint['training_epoch']
        self.inference_steps = checkpoint['inference_steps']

    @staticmethod
    def _build_placeholder_eval_obs(env_spawner: EnvSpawner) -> Dict[str, torch.Tensor]:
        """Returns a dictionary that mimics an evaluated observation with all values being 0.

        Parameters
        ----------
        env_spawner: :py:class:`.EnvSpawner`
            An :py:class:`.EnvSpawner` that holds information about the environment,
            that can be spawned.
        """
        placeholder_eval_obs = env_spawner.placeholder_obs
        placeholder_eval_obs['action'] = torch.zeros(1, 1)
        placeholder_eval_obs['baseline'] = torch.zeros(1, 1)
        placeholder_eval_obs['policy_logits'] = torch.zeros(
            1, 1, env_spawner.env_info['action_space'].n)
        placeholder_eval_obs['training_steps'] = torch.zeros(1, 1)

        return placeholder_eval_obs

    def _check_dead_queues(self, dead_threshold: int = 500):
        """Checks, if all queues has the same length for a chosen number of sequential times.

        If so, queues are assumed to be dead. The global shutdown is initiated in this case.

        Parameters
        ----------
        dead_threshold: `int`
            The maximum number of consecutive checks, if queues are dead.
        """
        if (self.queue_batches_old == self.queue_batches.qsize()) \
                and (self.queue_drop_off_old == self.queue_drops.qsize()) \
                and (self.queue_rpcs_old == len(self._pending_rpcs)):
            self.dead_counter += 1
        else:
            self.dead_counter = 0
            self.queue_batches_old = self.queue_batches.qsize()
            self.queue_drop_off_old = self.queue_drops.qsize()
            self.queue_rpcs_old = len(self._pending_rpcs)

        if self.dead_counter > dead_threshold:
            print("\n==========================================")
            print("CLOSING DUE TO DEAD QUEUES. (Used STRG+C?)")
            print("==========================================\n")
            self.shutdown = True

    def _cleanup(self, waiting_time=5):
        """Cleans up after main loop is done. Called by :py:meth:`~.RpcCallee.loop()`.

        Overwrites and calls :py:meth:`~.RpcCallee._cleanup()`.
        """
        self._save_model(self._model_path)

        self.runtime = self.get_runtime()
        self.queue_batches.close()
        self.queue_drops.close()
        self.trajectory_store.del_all()
        self.shutdown_event.set()

        super()._cleanup()

        # write last buffers
        print("Write and empty log buffers.")
        # pylint: disable=protected-access
        self.recorder._logger.write_buffers()

        print("Empty queues.")
        # Empty queues
        while self.queue_batches.qsize() > 0:
            batch = self.queue_batches.get(timeout=waiting_time)
            del batch
        while self.queue_drops.qsize() > 0:
            drop = self.queue_drops.get(timeout=waiting_time)
            del drop

        # Remove process to ensure freeing of resources.
        print("Join threads.")
        for thread in [*self.prefetch_threads, *self.storing_threads]:
            try:
                thread.join(timeout=waiting_time)
            except RuntimeError:
                # Timeout, thread died during shutdown
                pass

        self.queue_batches.join_thread()
        self.queue_drops.join_thread()

        print("Empty CUDA cache.")
        torch.cuda.empty_cache()

        # Run garbage collection to ensure freeing of resources.
        print("Running garbage collection.")
        gc.collect()

        if self._verbose:
            self._report()

    def _report(self):
        """Reports data to CLI
        """
        if self.runtime > 0:
            print("\n============== REPORT ==============")
            fps = self.inference_steps / self.runtime

            print("infered", str(self.inference_steps), "steps")
            print("in", str(self.runtime), "seconds")
            print("==>", str(fps), "fps")

            fps = self.training_steps / self.runtime
            print("trained", str(self.training_steps), "steps")
            print("in", str(self.runtime), "seconds")
            print("==>", str(fps), "fps")

            print("Total inference_time:", str(
                self.inference_time), "seconds")

            print("Total training_time:", str(
                self.training_time), "seconds")

            print("Total fetching_time:", str(
                self.fetching_time), "seconds")

            print("Mean inference latency:", str(
                self.recorder.mean_latency), "seconds")
