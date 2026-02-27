from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _QNetwork(nn.Module):
    """MLP that outputs (num_actions * num_objectives) values."""

    def __init__(self, obs_dim: int, num_actions: int, num_objectives: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions * num_objectives),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MODQN:
    """Multi-Objective DQN for continuous observation spaces.

    Outputs vector Q-values per action and uses a utility function
    for scalarized action selection.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_objectives: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.q_net = _QNetwork(obs_dim, num_actions, num_objectives)
        self.target_net = _QNetwork(obs_dim, num_actions, num_objectives)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self._buffer: deque[tuple] = deque(maxlen=buffer_size)
        self._step_count = 0

    def select_action(self, obs: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_flat = self.q_net(obs_t)
            q_values = q_flat.reshape(self.num_actions, self.num_objectives).numpy()

        if utility_fn is not None:
            scalarized = np.array([utility_fn(q_values[a]) for a in range(self.num_actions)])
        elif weights is not None:
            scalarized = q_values @ weights
        else:
            scalarized = q_values.sum(axis=1)

        return int(np.argmax(scalarized))

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
        self._buffer.append((obs.copy(), action, reward.copy(), next_obs.copy(), done))
        self._step_count += 1

        if len(self._buffer) < self.batch_size:
            return

        # Sample batch
        indices = self._rng.integers(0, len(self._buffer), size=self.batch_size)
        batch = [self._buffer[i] for i in indices]
        obs_b = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        act_b = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rew_b = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32)
        next_obs_b = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        done_b = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        # Current Q-values for taken actions: (batch, num_objectives)
        q_flat = self.q_net(obs_b)
        q_all = q_flat.reshape(self.batch_size, self.num_actions, self.num_objectives)
        q_current = q_all[torch.arange(self.batch_size), act_b]

        # Target: best next action via utility scalarization
        with torch.no_grad():
            next_q_flat = self.target_net(next_obs_b)
            next_q_all = next_q_flat.reshape(self.batch_size, self.num_actions, self.num_objectives)
            next_q_np = next_q_all.numpy()

            best_next_actions = np.zeros(self.batch_size, dtype=np.int64)
            for i in range(self.batch_size):
                if utility_fn is not None:
                    scores = np.array([utility_fn(next_q_np[i, a]) for a in range(self.num_actions)])
                elif weights is not None:
                    scores = next_q_np[i] @ weights
                else:
                    scores = next_q_np[i].sum(axis=1)
                best_next_actions[i] = np.argmax(scores)

            best_next = next_q_all[torch.arange(self.batch_size), torch.tensor(best_next_actions)]
            target = rew_b + self.gamma * (1 - done_b.unsqueeze(1)) * best_next

        loss = nn.functional.mse_loss(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
