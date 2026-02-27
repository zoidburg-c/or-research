from __future__ import annotations

from collections.abc import Callable

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class EnvelopeMOQ:
    """Wrapper around morl_baselines Envelope for multi-objective Q-learning.

    Creates a mock gymnasium env to satisfy Envelope's constructor,
    then exposes our standard agent API.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_objectives: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        learning_starts: int = 100,
        target_update_freq: int = 200,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self._default_weights = np.ones(num_objectives, dtype=np.float32) / num_objectives

        # Create mock env to satisfy Envelope constructor
        mock_env = _MockEnv(obs_dim, num_actions, num_objectives)

        from morl_baselines.multi_policy.envelope.envelope import Envelope

        self._agent = Envelope(
            env=mock_env,
            learning_rate=lr,
            initial_epsilon=epsilon,
            final_epsilon=epsilon,
            tau=1.0,
            target_net_update_freq=target_update_freq,
            buffer_size=buffer_size,
            net_arch=[128, 128],
            batch_size=batch_size,
            learning_starts=learning_starts,
            gamma=gamma,
            envelope=True,
            log=False,
            seed=seed,
        )

    def select_action(self, obs: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> int:
        w = self._resolve_weights(utility_fn, weights)
        return int(self._agent.eval(np.asarray(obs, dtype=np.float32), w))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: bool = False,
        utility_fn: Callable | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self._agent.replay_buffer.add(
            np.asarray(obs, dtype=np.float32),
            action,
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            done,
        )
        self._agent.global_step += 1
        if self._agent.global_step >= self._agent.learning_starts and len(self._agent.replay_buffer) >= self._agent.batch_size:
            self._agent.update()

    def _resolve_weights(self, utility_fn: Callable | None, weights: np.ndarray | None) -> np.ndarray:
        if weights is not None:
            return np.asarray(weights, dtype=np.float32)
        return self._default_weights.copy()


class _MockEnv(gym.Env):
    """Minimal gymnasium env to satisfy morl_baselines constructor."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, num_actions: int, num_objectives: int):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Discrete(num_actions)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float32)
        self.spec = None

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), np.zeros(self.reward_space.shape[0]), False, False, {}
