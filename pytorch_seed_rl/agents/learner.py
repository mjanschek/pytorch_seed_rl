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
"""
"""
import gc
import os
import pprint
import time
# import threading
from collections import deque
from typing import Dict, List, Tuple, Union

import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
# from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR

from .. import agents
from ..agents.rpc_callee import RpcCallee
from ..environments import EnvSpawner
from ..functional import loss, vtrace
from ..tools import Logger, TrajectoryStore
from ..tools.functions import listdict_to_dictlist


class Learner(RpcCallee):
    """Agent that runs inference and learning in parallel.

    This learning agent implements the reinforcement learning algorithm IMPALA
    following the SEED RL implementation by Google Brain.

    During initiation:
        * Spawns :py:attr:`num_actors` instances of :py:class:`~pytorch_seed_rl.agents.actor.Actor`.
        * Invokes their :py:meth:`~pytorch_seed_rl.agents.actor.Actor.loop()` methods.
        * Creates a :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        * Creates a :py:class:`~pytorch_seed_rl.tools.logger.Logger`.
        * Starts a continous prefetching process to prepare batches
          of complete trajectories for learning.

    During runtime:
        * Runs evaluates observations received from
          :py:class:`~pytorch_seed_rl.agents.Actor` and returns actions.
        * Stores incomplete trajectories in :py:class:`~pytorch_seed_rl.tools.trajectory_store`.
        * Trains a global model from trajectories received from a data prefetching thread.

    Parameters
    ----------
    rank : `int`
        Rank given by the RPC group on initiation (as in :py:func:`torch.distributed.rpc.init_rpc`).
    num_learners : `int`
        Number of total :py:class:`~pytorch_seed_rl.agents.learner.Learner`
        objects spawned by mother process.
    num_actors : `int`
        Number of total :py:class:`~pytorch_seed_rl.agents.actor.Actor` objects to spawn.
    env_spawner : :py:class:`~pytorch_seed_rl.environments.env_spawner.EnvSpawner`
        Object that spawns an environment on invoking it's
        :py:meth:`~pytorch_seed_rl.environments.env_spawner.EnvSpawner.spawn()` method.
    model : :py:class:`torch.nn.Module`
        A torch model that processes frames as returned
        by an environment spawned by :py:attr:`env_spawner`
    optimizer : :py:class:`torch.nn.Module`
        A torch optimizer that links to :py:attr:`model`
    inference_batchsize : `int`
        Number of RPCs to gather before running inferenve on their data.
    training_batchsize : `int`
        Number of complete trajectories to gather before learning from them as batch.
    rollout_length : `int`
        Length of rollout used by the IMPALA algorithm.
    max_epoch : `int`
        Maximum number of training epochs to do.
    total_steps : `int`
        Maximum number of environment steps to learn from.
    max_time : `int`
        Maximum time for training.
    save_path : `str`
        The root directory for saving data. Default: the current working directory.
    exp_name : `str`
        The title of the experiment to run.
        This creates a directory with this name in :py:attr:`save_dir`, if it does not exist.
    verbose : `bool`
        Set True if system metrics shall be printed at interval set by `print_interval`.
    print_interval : `int`
        Interval of training epoch system metrics shall be printed. Set to 0 to surpress printing.
    render: `bool`
        bla
    """

    def __init__(self,
                 rank: int,
                 num_learners: int,
                 num_actors: int,
                 env_spawner: EnvSpawner,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 total_steps: int = -1,
                 exp_name: str = "",
                 save_path: str = ".",
                 inference_batchsize: int = 4,
                 training_batchsize: int = 4,
                 rollout_length: int = 80,
                 max_epoch: int = -1,
                 max_time: float = -1.,
                 num_prefetchers: int = 1,
                 verbose: bool = False,
                 print_interval: int = 10,
                 render: bool = False,
                 max_training_batches: int = 128,
                 max_gif_length: int = 1024):

        self.total_num_envs = num_actors*env_spawner.num_envs
        self.envs_list = [i for i in range(self.total_num_envs)]

        super().__init__(rank,
                         num_callees=num_learners,
                         num_callers=num_actors,
                         caller_class=agents.Actor,
                         caller_args=[env_spawner],
                         future_keys=self.envs_list,
                         rpc_batchsize=inference_batchsize)

        # ATTRIBUTES
        self.rollout_length = rollout_length
        self.rpc_batchsize = inference_batchsize
        self.training_batchsize = training_batchsize
        self.max_epoch = max_epoch
        self.total_steps = total_steps
        self.max_time = max_time
        self.save_path = save_path
        self.exp_name = exp_name
        self.verbose = verbose
        self.print_interval = print_interval
        self.render = render
        self.max_gif_length = max_gif_length

        # storage
        self.training_batch_queue = deque(
            maxlen=max_training_batches
        )
        self.states_to_store = {}
        self.rec_frames = []

        # counters
        self.inference_epoch = 0
        self.inference_steps = 0
        self.inference_time = 0.

        self.training_epoch = 0
        self.training_steps = 0
        self.training_time = 0.

        self.fetching_time = 0.

        self.runtime = 0
        self.mean_latency = 0.
        self.episodes_seen = 0
        self.trajectories_seen = 0
        self.record_eps_id = None
        self.return_memory = 0

        self.best_return = 0

        self.last_time = 0.
        self.dead_counter = 0

        # torch
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)

        self.model = model.to(self.device)
        self.eval_model = self.model
        self.eval_model.eval()

        self.optimizer = optimizer

        # define a linear decreasing function for linear scheduler
        def linear_lambda(epoch):
            return 1 - min(epoch * rollout_length * training_batchsize, total_steps) / total_steps
        self.scheduler = LambdaLR(self.optimizer, linear_lambda)

        # Create thread locks
        self.lock_model = mp.Lock()
        self.lock_prefetch = mp.Lock()
        self.lock_record = mp.Lock()
        self.lock_store = mp.Lock()

        # spawn trajectory store
        placeholder_eval_obs = self._build_placeholder_eval_obs(env_spawner)
        self.trajectory_store = TrajectoryStore(self.envs_list,
                                                placeholder_eval_obs,
                                                self.device,
                                                max_trajectory_length=rollout_length)

        # spawn logger
        self.logger = Logger(['episodes', 'training', 'system'],
                             "/".join([self.save_path, self.exp_name]),
                             modes=['csv'])

        # start prefetch threads as remote rpc
        self.prefetch_threads = [self.rref.remote().prefetch()
                                 for _ in range(num_prefetchers)]

        self.queue_batches_old = len(self.training_batch_queue)
        self.queue_drop_off_old = len(self.trajectory_store.drop_off_queue)
        self.queue_rpcs_old = len(self.pending_rpcs)

        # start callers
        self._start_callers()

    def _loop(self,
              sleep_time: float = 0.1,
              system_log_interval: int = 1):
        """Inner loop function of a :py:class:`Learner`.

        Called by :py:meth:`loop()`.

        This method first waits on :py:attr:`self.event_start_training`.
        Then it invokes :py:meth:`_learn_from_batch()`
        and copies the updated model weights from the learning model to :py:attr:`self.eval_model`.
        System metrics are passed to :py:attr:`self.logger`.
        Finally, it checks for reached shutdown criteria, like :py:attr:`self.total_steps` has been reached.
        """

        if len(self.training_batch_queue) > 0:
            start = time.time()
            batch = self.training_batch_queue.popleft()
            training_metrics = self._learn_from_batch(batch,
                                                      grad_norm_clipping=40.,
                                                      pg_cost=1.,
                                                      baseline_cost=0.5,
                                                      entropy_cost=0.01)
            self.training_time += time.time()-start

            # parameters are shared between inference and training model automatically

            self.logger.log('training', training_metrics)
        else:
            time.sleep(sleep_time)

        if self._loop_iteration == system_log_interval:
            system_metrics = self._get_system_metrics()
            self.logger.log('system', system_metrics)
            self._loop_iteration = 0

        if self.verbose and (self.training_epoch % self.print_interval == 0):
            print(pprint.pformat(system_metrics))

        # check if queues are dead
        self._check_dead_queues()

        # check, if shutdown prerequisites haven been reached
        self.shutdown = ((self.training_epoch > self.max_epoch > 0) or
                         (self.training_steps > self.total_steps > 0) or
                         (self._get_runtime() > self.max_time > 0) or
                         self.shutdown)

    def _check_dead_queues(self, dead_threshold=100):
        """Checks, if all queues has the same length for a chosen number of sequential times.

        If so, queues are assumed to be dead. The global shutdown is initiated in this case.
        """
        if (self.queue_batches_old == len(self.training_batch_queue)) \
                and (self.queue_drop_off_old == len(self.trajectory_store.drop_off_queue)) \
                and (self.queue_rpcs_old == len(self.pending_rpcs)):
            self.dead_counter += 1
        else:
            self.dead_counter = 0
            self.queue_batches_old = len(self.training_batch_queue)
            self.queue_drop_off_old = len(self.trajectory_store.drop_off_queue)
            self.queue_rpcs_old = len(self.pending_rpcs)

        if self.dead_counter > dead_threshold:
            print("\n==========================================")
            print("CLOSING DUE TO DEAD QUEUES. (Used STRG+C?)")
            print("==========================================\n")
            self.shutdown = True

    @staticmethod
    def _build_placeholder_eval_obs(env_spawner: EnvSpawner) -> Dict[str, torch.Tensor]:
        """Returns a dictionary that mimics an evaluated observation with all values being 0.
        """
        placeholder_eval_obs = env_spawner.placeholder_obs
        placeholder_eval_obs['action'] = torch.zeros(1, 1)
        placeholder_eval_obs['baseline'] = torch.zeros(1, 1)
        placeholder_eval_obs['policy_logits'] = torch.zeros(
            1, 1, env_spawner.env_info['action_space'].n)

        return placeholder_eval_obs

    def process_batch(self,
                      caller_ids: List[Union[int, str]],
                      *batch: List[dict],
                      **misc: dict) -> Dict[str, torch.Tensor]:
        """Inner method to process a whole batch at once.

        Called by :py:meth:`_process_batch()`.

        Before returning the result for the given batch, this method:
            #. Moves its data to the :py:class:`Learner` device (usually GPU)
            #. Runs inference on this data
            #. Sends evaluated data to
               :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`
               using a parallel RPC of :py:meth:`add_to_store()`.

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
        for b in batch:
            for k, v in b.items():
                try:
                    # [T, B, C, H, W] => [1, batchsize, C, H, W]
                    b[k] = torch.cat(v, dim=1).to(self.device)
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

        self.inference_steps += self.rpc_batchsize
        self.inference_epoch += 1

        # add states to store in parallel process. Don't move data via RPC as it shall stay on cuda.
        states_to_store = {k: v.detach()
                           for k, v in {**states, **inference_output}.items()}
        self.add_to_store(states_to_store, caller_ids, misc['metrics'])

        # gather an return results
        results = {c: inference_output['action'][0][i].view(
            1, 1).cpu().detach() for i, c in enumerate(caller_ids)}

        return results

    def add_to_store(self,
                     states,
                     caller_ids: List[Union[int, str]],
                     all_metrics: dict):
        """Sends states within :py:attr:`self.states_to_store` and metrics to
        :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`
        according to :py:attr:`caller_ids`.

        Parameters
        ----------
        caller_ids : `list[int]` or `list[str]`
            List of unique identifiers for callers.
        all_metrics : `dict`
            Recorded metrics of these states
            (primarily recorded in :py:class:`~pytorch_seed_rl.agents.actor.Actor`.)
        """
        # pylint: disable=not-callable
        timestamp = torch.tensor(time.time(), dtype=torch.float64).view(1, 1)

        # with self.lock_store:
        # states = self.states_to_store

        # extract single states and send to trajectory store
        for i, zipped_caller_id_metrics in enumerate(list(zip(caller_ids, all_metrics))):
            caller_id, metrics = zipped_caller_id_metrics

            state = {k: v[0, i] for k, v in states.items()}
            metrics['timestamp'] = timestamp.clone()

            self.trajectory_store.add_to_entry(caller_id, state, metrics)

    def _learn_from_batch(self,
                          batch: Dict[str, torch.Tensor],
                          grad_norm_clipping: float = 40.,
                          pg_cost: float = 1.,
                          baseline_cost: float = 0.5,
                          entropy_cost: float = 0.01):
        """Runs the learning process and updates the internal model.

        This method:
            #. Evaluates the given :py:attr:`batch` with the internal learning model.
            #. Invokes :py:meth:`compute_losses()` to get all components of the loss function.
            #. Calculates the total loss, using the given cost factors for each component.
            #. Updates the model by invoking the :py:attr:`self.optimizer`.

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
            batch, learner_outputs)

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

        # return metrics
        return {"runtime": self._get_runtime(),
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
        * :py:mod:`~pytorch_seed_rl.functional.loss`
        * :py:mod:`~pytorch_seed_rl.functional.vtrace`

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

        # print(vtrace_returns.vs[:,0])

        pg_loss = loss.policy_gradient(learner_outputs["policy_logits"],
                                       batch["action"],
                                       vtrace_returns.pg_advantages)

        baseline_loss = F.mse_loss(learner_outputs["baseline"],
                                   vtrace_returns.vs,
                                   reduction='sum')

        entropy_loss = loss.entropy(learner_outputs["policy_logits"])

        return pg_loss, baseline_loss, entropy_loss

    def prefetch(self, waiting_time=0.1, max_tries=50):
        """Continuously prefetches complete trajectories dropped by
        the :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore` for training.

        As long as shutdown is not set, this method checks,
        if :py:attr:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore.drop_off_queue`
        has at least :py:attr:`self.training_batchsize` elements.
        If so, these trajectories are popped from this queue, logged,
        transformed and queued in :py:attr:`self.training_batch_queue`.

        This usually runs as asynchronous process.

        Parameters
        ----------
        waiting_time: `float`
            Time the methods loop sleeps between each iteration.
        """

        while not self.shutdown:

            trajectories = []
            start = time.time()
            with self.lock_prefetch:
                if len(self.trajectory_store.drop_off_queue) >= self.training_batchsize:
                    for _ in range(self.training_batchsize):
                        t = self.trajectory_store.drop_off_queue.popleft()
                        try:
                            self.log_trajectory(t)
                        except Exception as e:
                            print(e)
                        trajectories.append(t)

            if len(trajectories) > 0:
                batch = self._to_batch(trajectories)

                counter = 0

                if len(self.training_batch_queue) < self.training_batch_queue.maxlen:
                    self.training_batch_queue.append(batch)
                elif counter < max_tries:
                    counter += 1
                    time.sleep(waiting_time)
                else:
                    # essentially drop this batch
                    return

                # update stats
                self.fetching_time += time.time() - start
            else:
                time.sleep(waiting_time)

    def _cleanup(self):
        """Cleans up after main loop is done. Called by :py:meth:`loop()`

        Overwrites and calls :py:meth:`~pytorch_seed_rl.agents.rpc_callee.RpcCallee._cleanup()`.
        """
        self.runtime = self._get_runtime()

        super()._cleanup()

        # write last buffers
        print("Write and empty log buffers.")
        self.logger.write_buffers()

        # Remove process to ensure freeing of resources.
        print("Join prefetch threads.")
        for p in self.prefetch_threads:
            try:
                p.to_here(5)
            except RuntimeError:
                pass
            # Timeout, prefetch_thread died during shutdown

        # Run garbage collection to ensure freeing of resources.
        print("Running garbage collection.")
        gc.collect()

        self._report()

    def log_trajectory(self, trajectory: dict):
        """Extracts and logs episode data from a completed trajectory.

        Parameters
        ----------
        trajectory: `dict`
            Trajectory dropped by
            :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        """
        self.trajectories_seen += 1

        i_start = 0
        # iterate through trajectory
        for eps_id in trajectory['states']['episode_id'].unique():
            # find break point
            for i, next_id in enumerate(trajectory['states']['episode_id'][i_start:]):
                i_end = i_start + i
                if eps_id != next_id:
                    break
            # done is set next state
            done = trajectory['states']['done'][i_end].item()

            steps = trajectory['states']['episode_step'][i_end-1]
            if steps > 1000:
                print(eps_id, steps)

            if done:
                self.log_episode(trajectory, i_start, i_end)

            if self.render:
                self.record_trajectory(
                    trajectory, i_start, i_end, done)

            i_start = i_end

    def log_episode(self,
                    trajectory: dict,
                    i_start: int,
                    i_end: int):
        """Extracts and logs episode data from a completed trajectory.

        Parameters
        ----------
        trajectory: `dict`
            Trajectory dropped by
            :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        trajectory_end: `int`
        """
        self.episodes_seen += 1

        state = {k: v[i_end-1] for k, v in trajectory['states'].items()}
        metrics = {k: v[i_start:i_end+1]
                   for k, v in trajectory['metrics'].items()}

        mean_latency = metrics['latency'].mean().item()
        self.mean_latency = self.mean_latency + \
            (mean_latency - self.mean_latency) / self.episodes_seen

        episode_data = {
            'episode_id': state['episode_id'],
            'return': state['episode_return'],
            'length': state['episode_step'],
            'training_steps': state['training_steps'],
        }

        self.logger.log('episodes', episode_data)

    def record_trajectory(self,
                          trajectory: dict,
                          i_start: int,
                          i_end: int,
                          done: bool,
                          max_timediff: float = 300,
                          max_len: int = 0):

        if max_len <= 0:
            max_len = self.max_gif_length

        done_incr = int(not done)

        eps_id = trajectory['states']['episode_id'][i_start].item()

        current_time = self._get_runtime()
        time_diff = (current_time - self.last_time)

        # i_start > 0 can miss episodes, however it ensures that new records always start with episode start
        if self.record_eps_id is None and i_start > 0:
            self.record_eps_id = eps_id
            print("Start saving frames for episode %s" % eps_id)
        elif self.record_eps_id != eps_id:
            # reset if timeout
            if time_diff > max_timediff:
                print("Reset episode %s due to timeout." % eps_id)
                self.record_eps_id = None
                self.rec_frames = []
            return

        self.last_time = current_time

        # watch out for stacked frames
        frames = trajectory['states']['frame'][i_start:(i_end+done_incr), 0, 0].clone(
        ).to('cpu').numpy()

        for f in frames:
            # skip black screens (should not happen)
            if np.sum(f) > 0:
                if len(self.rec_frames) > 0:
                    # skip frame, if it did not change
                    if not np.array_equal(f, self.rec_frames[-1]):
                        self.rec_frames.append(f)
                else:
                    self.rec_frames.append(f)

        if done:
            eps_tsteps = trajectory['states']['training_steps'][i_end-1].item()
            eps_return = trajectory['states']['episode_return'][i_end-1].item()
            if eps_return > self.best_return:
                self.best_return = eps_return
                fname = "e%d_r%d_t%d" % (eps_id, eps_return, eps_tsteps)
                self._record_episode(self.rec_frames, fname)

            self.record_eps_id = None
            self.rec_frames = []

    def _record_episode(self, frames, filename):
        """
        """
        print("Record file %s.gif with %d frames!" % (filename, len(frames)))
        rec_array = np.asarray(frames, dtype='uint8')
        # [T, H, W]

        directory = "/".join([self.save_path, self.exp_name, 'gif'])
        os.makedirs(directory, exist_ok=True)
        imageio.mimsave(directory + '/%s.gif' % filename, rec_array, fps=20)

    def _to_batch(self, trajectories: List[dict]):
        """Extracts states from a list of trajectories, returns them as batch.

        Parameters
        ----------
        trajectories: `list`
            List of trajectories dropped by
            :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        """
        states = listdict_to_dictlist([t['states'] for t in trajectories])

        for k, v in states.items():
            # [T, B, C, H, W]  => [len(trajectories), batchsize, C, H, W]
            states[k] = torch.cat(v, dim=1).to(self.device)

        states['current_length'] = torch.stack(
            [t['current_length'] for t in trajectories])

        return states

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
                self.mean_latency), "seconds")

    def _get_system_metrics(self):
        """Returns the training systems metrics.
        """
        return {
            "runtime": self._get_runtime(),
            "trajectories_seen": self.trajectories_seen,
            "episodes_seen": self.episodes_seen,
            "mean_inference_latency": self.mean_latency,
            "fetching_time": self.fetching_time,
            "inference_time": self.inference_time,
            "inference_steps": self.inference_steps,
            "training_time": self.training_time,
            "training_steps": self.training_steps,
            "queue_batches": len(self.training_batch_queue),
            "queue_drop_off": len(self.trajectory_store.drop_off_queue),
            "queue_rpcs": len(self.pending_rpcs)
        }
