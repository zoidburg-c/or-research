from __future__ import annotations

import numpy as np


class TabularMOQLearning:
    """Multi-Objective Q-Learning with vector-valued Q-table.

    Uses scalarized action selection (weight vector applied at decision time)
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

    def select_action(self, state: int, weights: np.ndarray) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        scalarized = self.q_table[state] @ weights
        return int(np.argmax(scalarized))

    def update(
        self,
        state: int,
        action: int,
        reward: np.ndarray,
        next_state: int,
        weights: np.ndarray,
    ) -> None:
        scalarized_next = self.q_table[next_state] @ weights
        best_next_action = int(np.argmax(scalarized_next))
        target = reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def get_pareto_q_values(self, state: int) -> np.ndarray:
        return self.q_table[state].copy()
