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

# pylint: disable=not-callable, empty-docstring
"""
"""
import os
from typing import List

import imageio
import numpy as np
import torch

from .logger import Logger


class Recorder():
    """
    """

    def __init__(self,
                 save_path='',
                 render=False,
                 max_gif_length=10000):
        # ATTRIBUTES
        self.save_path = save_path
        self.render = render
        self.max_gif_length = max_gif_length
        self._logger = Logger(['episodes', 'training', 'system'],
                              self.save_path,
                              modes=['csv'])
        # COUNTERS
        self.episodes_seen = 0
        self.trajectories_seen = 0

        # STORAGE
        self.mean_latency = 0.
        self.rec_frames = []
        self.record_eps_id = None
        self.best_return = None
        self.record_return = 0

    def log(self, key, in_data):
        """Wrapps :py:meth:`Logger.log()`.
        """
        self._logger.log(key, in_data)

    def log_trajectory(self, trajectory: dict):
        """Extracts and logs episode data from a completed trajectory.

        Parameters
        ----------
        trajectory: `dict`
            Trajectory dropped by
            :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        """
        self.trajectories_seen += 1

        if not (self.record_eps_id is None or
                self.record_eps_id in trajectory['states']['episode_id'] or
                self.record_eps_id in trajectory['states']['prev_episode_id'] or
                True in trajectory['states']['done']):
            return

        # iterate through trajectory
        for i, done in enumerate(trajectory['states']['done']):

            # find break point
            if done and i > 0:
                self._log_episode(trajectory, i-1)

                if self.render and trajectory['states']['prev_episode_id'][i] == self.record_eps_id:
                    self._record_episode()

            if self.render:
                eps_id = trajectory['states']['episode_id'][i]
                if self.record_eps_id is None:
                    self.record_eps_id = eps_id
                if eps_id == self.record_eps_id:
                    self.record_return = trajectory['states']['episode_return'][i]
                    self._record_frame(trajectory['states']['frame'][i])

            # drop recorded data if:
            #   - saved buffer grows too long
            #   - or current episode_id is 1000 episodes higher
            #   - or episode runs very long (which can happen due to bugs of env)
            if self.record_eps_id is not None:
                if ((0 < self.max_gif_length <= len(self.rec_frames)) or
                        (trajectory['states']['episode_id'][i] - self.record_eps_id > 1000) or
                        (trajectory['states']['episode_step'][i] > 10*60*24)):

                    self.record_eps_id = None
                    self.rec_frames = []

    def _log_episode(self,
                     trajectory: dict,
                     i: int):
        """Extracts and logs episode data from a completed trajectory.

        Parameters
        ----------
        trajectory: `dict`
            Trajectory dropped by
            :py:class:`~pytorch_seed_rl.tools.trajectory_store.TrajectoryStore`.
        trajectory_end: `int`
        """
        self.episodes_seen += 1

        state = {k: v[i] for k, v in trajectory['states'].items()}
        metrics = {k: v[i] for k, v in trajectory['metrics'].items()}

        latency = metrics['latency'].item()
        self.mean_latency = self.mean_latency + \
            (latency - self.mean_latency) / self.episodes_seen

        episode_data = {
            'episode_id': state['episode_id'],
            'return': state['episode_return'],
            'length': state['episode_step'],
            'training_steps': state['training_steps'],
        }

        self._logger.log('episodes', episode_data)

    def _record_frame(self, frame: torch.Tensor):
        """
        """
        frame = frame[0, 0].clone().to('cpu').numpy()

        # skip black screens (should not happen)
        if np.sum(frame) > 0:
            if len(self.rec_frames) > 0:
                # skip frame, if it did not change
                if not np.array_equal(frame, self.rec_frames[-1]):
                    self.rec_frames.append(frame)
            else:
                self.rec_frames.append(frame)

    def _record_episode(self):
        """Empties :py:attr:`self.rec_frames` and writes a gif, if episode score is a new record.

        If :py:attr:`self.record_return` is a new record, write gif file.
        """
        if self.best_return is None or self.record_return > self.best_return:
            print("Record eps %d with %d frames and %f return!" %
                  (self.record_eps_id, len(self.rec_frames), self.record_return))
            self.best_return = self.record_return

            fname = "e%d_r%d" % (self.record_eps_id, self.record_return)
            self._write_gif(self.rec_frames, fname)
        self.record_eps_id = None
        self.rec_frames = []

    def _write_gif(self,
                   frames: List[np.ndarray],
                   filename: str):
        """Writes a gif file made off the list of :py:attr:`frames`
        using the given :py:attr:`filename`.

        Parameters
        ----------
        frames: `list` of py:obj:`numpy.ndarray`
            A list of images as numpy arrays.
        filename: `str`
            A name for the created gif file.
        """
        rec_array = np.asarray(frames, dtype='uint8')
        # [T, H, W]

        directory = os.path.join(self.save_path, 'gif')
        os.makedirs(directory, exist_ok=True)

        fpath = os.path.join(directory, '%s.gif' % filename)
        imageio.mimsave(fpath, rec_array, fps=20)
