# tests/test_agents.py
import numpy as np
import pytest
from agents.mo_q_learning import TabularMOQLearning


class TestTabularMOQLearning:
    def test_init(self):
        agent = TabularMOQLearning(
            num_states=10,
            num_actions=4,
            num_objectives=3,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=0.3,
            seed=42,
        )
        assert agent.q_table.shape == (10, 4, 3)

    def test_select_action(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2, epsilon=0.0, seed=42,
        )
        agent.q_table[0, 1, :] = [10.0, 10.0]
        action = agent.select_action(state=0, weights=np.array([0.5, 0.5]))
        assert action == 1

    def test_update(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2,
            learning_rate=1.0, gamma=0.0, epsilon=0.0, seed=42,
        )
        agent.update(
            state=0, action=1, reward=np.array([5.0, 3.0]),
            next_state=1, weights=np.array([0.5, 0.5]),
        )
        np.testing.assert_array_almost_equal(agent.q_table[0, 1], [5.0, 3.0])

    def test_epsilon_exploration(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2, epsilon=1.0, seed=42,
        )
        actions = [agent.select_action(0, np.array([0.5, 0.5])) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1
