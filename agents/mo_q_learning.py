from __future__ import annotations

from collections.abc import Callable

import numpy as np


class TabularMOQLearning:
    """Multi-Objective Q-Learning with vector-valued Q-table.

    Uses scalarized action selection (utility function applied at decision time)
    but maintains the full vector Q-values for multi-objective analysis.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_objectives: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

        self.q_table = np.zeros((num_states, num_actions, num_objectives), dtype=np.float64)

    def _scalarize(self, q_values: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> np.ndarray:
        """Scalarize Q-values for all actions using utility_fn or weights."""
        if utility_fn is not None:
            return np.array([utility_fn(q_values[a]) for a in range(q_values.shape[0])])
        return q_values @ weights

    def select_action(self, state: int, weights: np.ndarray | None = None, utility_fn: Callable | None = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        scalarized = self._scalarize(self.q_table[state], utility_fn, weights)
        return int(np.argmax(scalarized))

    def update(
        self,
        state: int,
        action: int,
        reward: np.ndarray,
        next_state: int,
        weights: np.ndarray | None = None,
        utility_fn: Callable | None = None,
    ) -> None:
        scalarized_next = self._scalarize(self.q_table[next_state], utility_fn, weights)
        best_next_action = int(np.argmax(scalarized_next))
        target = reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def get_pareto_q_values(self, state: int) -> np.ndarray:
        return self.q_table[state].copy()
