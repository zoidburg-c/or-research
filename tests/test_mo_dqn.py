import numpy as np
import pytest
import torch
from agents.mo_dqn import MODQN
from momas.utility import linear_utility, threshold_utility


class TestMODQN:
    def test_init(self):
        agent = MODQN(obs_dim=12, num_actions=7, num_objectives=3, seed=42)
        assert agent.obs_dim == 12
        assert agent.num_actions == 7
        assert agent.num_objectives == 3

    def test_select_action_shape(self):
        agent = MODQN(obs_dim=12, num_actions=7, num_objectives=3, epsilon=0.0, seed=42)
        obs = np.random.default_rng(42).standard_normal(12).astype(np.float32)
        u = linear_utility(np.array([1/3, 1/3, 1/3]))
        action = agent.select_action(obs, utility_fn=u)
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_select_action_with_different_utilities(self):
        agent = MODQN(obs_dim=4, num_actions=3, num_objectives=2, epsilon=0.0, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        u1 = linear_utility(np.array([0.9, 0.1]))
        u2 = linear_utility(np.array([0.1, 0.9]))
        a1 = agent.select_action(obs, utility_fn=u1)
        a2 = agent.select_action(obs, utility_fn=u2)
        assert 0 <= a1 < 3
        assert 0 <= a2 < 3

    def test_epsilon_exploration(self):
        agent = MODQN(obs_dim=4, num_actions=5, num_objectives=2, epsilon=1.0, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        u = linear_utility(np.array([0.5, 0.5]))
        actions = [agent.select_action(obs, utility_fn=u) for _ in range(100)]
        assert len(set(actions)) > 1

    def test_update_runs_without_error(self):
        agent = MODQN(obs_dim=4, num_actions=3, num_objectives=2,
                      buffer_size=100, batch_size=4, seed=42)
        u = linear_utility(np.array([0.5, 0.5]))
        obs = np.zeros(4, dtype=np.float32)
        for i in range(10):
            next_obs = np.ones(4, dtype=np.float32) * i
            agent.update(obs, i % 3, np.array([1.0, 2.0]), next_obs, done=False, utility_fn=u)
            obs = next_obs

    def test_update_changes_q_values(self):
        agent = MODQN(obs_dim=4, num_actions=3, num_objectives=2,
                      buffer_size=100, batch_size=4, lr=0.01, seed=42)
        u = linear_utility(np.array([0.5, 0.5]))
        obs = np.zeros(4, dtype=np.float32)

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_before = agent.q_net(obs_t).reshape(agent.num_actions, agent.num_objectives).clone()

        for i in range(20):
            agent.update(obs, 0, np.array([10.0, 10.0]), obs, done=False, utility_fn=u)

        with torch.no_grad():
            q_after = agent.q_net(obs_t).reshape(agent.num_actions, agent.num_objectives)

        assert not torch.allclose(q_before, q_after)
