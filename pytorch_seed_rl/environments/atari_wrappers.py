# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Taken from
#   https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/atari_wrappers.py
# and slightly modified (again)

# pylint: disable=missing-module-docstring, missing-class-docstring
# pylint: missing-function-docstring, too-many-arguments, arguments-differ
"""A collection of wrappers applicable to environments following the OpenAI gym API

See Also
--------
`OpenAI Gym <https://gym.openai.com/>`__
"""

from collections import deque

import cv2
import gym
import numpy as np
import torch
from gym import spaces

cv2.ocl.setUseOpenCL(False)


def make_atari(env_id: str) -> gym.Env:
    """Creates the :py:class:`~gym.Env` registered with `gym <https://gym.openai.com/docs/>`__.

    Accepts only environments that don't perform frameskip natively.

    Always applies:
        * :py:class:`NoopResetEnv` (`noop_max` = 30)
        * :py:class:`MaxAndSkipEnv` (`skip` = 4)

    Arguments
    ---------
    env_id: `str`
        The environments identifier as registered with :py:mod:`gym`.
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    return env


def wrap_deepmind(env,
                  episode_life: bool = True,
                  clip_rewards: bool = True,
                  frame_stack: bool = False,
                  scale: bool = False) -> gym.Env:
    """Configure environment for DeepMind-style Atari.

    Always applies:
        * :py:class:`WarpFrame`
        * :py:class:`FireResetEnv`, if :py:attr:`env` contains an action with meaning 'FIRE'

    Arguments
    ---------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    episode_life: `bool`
        Applies :py:class:`EpisodicLifeEnv`, if True.
    clip_rewards: `bool`
        Applies :py:class:`ClipRewardEnv`, if True.
    frame_stack: `bool`
        Applies :py:class:`FrameStack` (`k` = 4), if True.
    scale: `bool`
        Applies :py:class:`ScaledFloatFrame`, if True.
    """
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    return env


def wrap_pytorch(env) -> gym.Env:
    """Applies :py:class:`ImageToPyTorch` as wrap.

    Arguments
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """
    return ImageToPyTorch(env)


class LazyFrames():
    """This object ensures that common frames between the observations are only stored once.

    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was.

    Parameters
    ----------
    frames: `list`
        A list of frames that shall be converted.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        # pylint: disable=unsubscriptable-object

        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class AutoResetWrapper(gym.Wrapper):
    """A wrapper that automatically resets the environment in case of termination.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._terminated = False

    def step(self, action):
        if self._terminated:
            self.env.reset()
        observation, reward, terminal, info = self.env.step(action)
        self._terminated = terminal
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._terminated = False
        return observation


class ClipRewardEnv(gym.RewardWrapper):
    """Clips rewards.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class DictObservationsEnv(gym.Wrapper):
    """Provides observations as `dict` with additional metrics.

    Adds :py:meth:`initial()` method, which returns the initial observation.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_return = None
        self.episode_step = None

    def initial(self) -> dict:
        """Returns an initial observation.
        """
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        initial_done = torch.zeros(1, 1, dtype=torch.bool)
        initial_frame = self.reset()

        obs = dict(frame=initial_frame,
                   reward=initial_reward,
                   done=initial_done,
                   episode_return=self.episode_return,
                   episode_step=self.episode_step,
                   last_action=initial_last_action
                   )
        try:
            # pylint: disable=not-callable
            obs['real_done'] = torch.tensor(
                self.was_real_done, dtype=torch.bool).view(1, 1)
        except AttributeError:
            pass
        return obs

    def step(self, action):
        frame, reward, done, unused_info = self.env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.reset()

        # pylint: disable=not-callable
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        obs = dict(frame=frame,
                   reward=reward,
                   done=done,
                   episode_return=episode_return,
                   episode_step=episode_step,
                   last_action=action,
                   )
        try:
            # pylint: disable=not-callable
            obs['real_done'] = torch.tensor(
                self.was_real_done, dtype=torch.bool).view(1, 1)
        except AttributeError:
            pass

        return obs

    def reset(self, **kwargs):
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        return self.env.reset(**kwargs)

    def close(self):
        self.env.close()


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FrameStack(gym.Wrapper):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.

    See Also
    --------
    :py:class:`LazyFrames`

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    k: `int`
        Number of last frames to stack.
    """

    def __init__(self,
                 env: gym.Env,
                 k: int):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ImageToPyTorch(gym.ObservationWrapper):
    """Changes image shape to channels x weight x height

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return self._format_frame(
            np.transpose(observation, axes=(2, 0, 1)))

    @staticmethod
    def _format_frame(frame):
        frame = torch.from_numpy(frame)
        return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    skip: `int`
        The number of the returned frame. If `skip` = 4 (default),
        only every 4th frame will be returned.
    """

    def __init__(self,
                 env: gym.Env,
                 skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    noop_max: `int`
        The maximum number of no-ops on reset.
    """

    def __init__(self,
                 env: gym.Env,
                 noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalizes the frame.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class WarpFrame(gym.ObservationWrapper):
    """ Warp frames to `height`x`width` as done in the Nature paper and later work.
    If the environment uses dictionary observations, `dict_space_key`
    can be specified which indicates which
    observation should be warped.

    Parameters
    ----------
    env: :py:obj:`gym.Env`
        An environment that will be wrapped.
    width: `int`
        Target width of warped frames.
    height: `int`
        Target height of warped frames.
    grayscale: `bool`
        Set True,. if warped frames shall be greyscale.
    dict_space_key: `str`
        Key of targeted space of environments observation space dictionary.

    """

    def __init__(self,
                 env: gym.Env,
                 width: int = 84,
                 height: int = 84,
                 grayscale: bool = True,
                 dict_space_key: str = None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

    def observation(self, observation):
        if self._key is None:
            frame = observation
        else:
            frame = observation[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            observation = frame
        else:
            observation = observation.copy()
            observation[self._key] = frame
        return observation
