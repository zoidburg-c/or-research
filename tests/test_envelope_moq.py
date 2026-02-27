import numpy as np
import pytest
from agents.envelope_moq import EnvelopeMOQ
from momas.utility import linear_utility


class TestEnvelopeMOQ:
    def test_init(self):
        agent = EnvelopeMOQ(obs_dim=12, num_actions=7, num_objectives=3, seed=42)
        assert agent.obs_dim == 12
        assert agent.num_actions == 7
        assert agent.num_objectives == 3

    def test_select_action_shape(self):
        agent = EnvelopeMOQ(obs_dim=12, num_actions=7, num_objectives=3, seed=42)
        obs = np.random.default_rng(42).standard_normal(12).astype(np.float32)
        u = linear_utility(np.array([1/3, 1/3, 1/3]))
        action = agent.select_action(obs, utility_fn=u)
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_select_action_with_weights(self):
        agent = EnvelopeMOQ(obs_dim=4, num_actions=3, num_objectives=2, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        w = np.array([0.5, 0.5])
        action = agent.select_action(obs, weights=w)
        assert 0 <= action < 3

    def test_update_runs_without_error(self):
        agent = EnvelopeMOQ(obs_dim=4, num_actions=3, num_objectives=2,
                            buffer_size=100, batch_size=4, learning_starts=5, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        w = np.array([0.5, 0.5])
        for i in range(10):
            next_obs = np.ones(4, dtype=np.float32) * i * 0.1
            agent.update(obs, i % 3, np.array([1.0, 2.0]), next_obs, done=False, weights=w)
            obs = next_obs
