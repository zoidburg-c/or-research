import functools
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from momas.base_env import MOMASConfig, MOMASWrapper
from momas.utility import linear_utility


class StubEnv(ParallelEnv):
    """Minimal 2-agent, 2-objective environment for testing."""

    metadata = {"name": "stub_v0"}

    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = []
        self._rewards = {
            "agent_0": np.array([10.0, 20.0]),
            "agent_1": np.array([30.0, 40.0]),
        }
        self._team_reward = np.array([50.0, 60.0])
        self._step_count = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(4,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._step_count = 0
        obs = {a: np.zeros(4, dtype=np.float32) for a in self.agents}
        infos = {a: {"vec_reward": self._rewards[a]} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self._step_count += 1
        obs = {a: np.zeros(4, dtype=np.float32) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        done = self._step_count >= 5
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {
            a: {
                "vec_reward": self._rewards[a],
                "team_vec_reward": self._team_reward,
            }
            for a in self.agents
        }
        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos


class TestMOMASConfig:
    def test_valid_config(self):
        cfg = MOMASConfig(
            reward_structure="team",
            utility_type="team",
            optimisation_criterion="SER",
            num_objectives=2,
            utility_functions={"shared": linear_utility(np.array([0.5, 0.5]))},
        )
        assert cfg.reward_structure == "team"

    def test_invalid_reward_individual_utility_team(self):
        with pytest.raises(ValueError, match="not a meaningful combination"):
            MOMASConfig(
                reward_structure="individual",
                utility_type="team",
                optimisation_criterion="SER",
                num_objectives=2,
                utility_functions={"shared": linear_utility(np.array([0.5, 0.5]))},
            )


class TestMOMASWrapperTeamReward:
    def setup_method(self):
        self.env = StubEnv()
        self.cfg = MOMASConfig(
            reward_structure="team",
            utility_type="team",
            optimisation_criterion="SER",
            num_objectives=2,
            utility_functions={"shared": linear_utility(np.array([0.5, 0.5]))},
        )
        self.wrapped = MOMASWrapper(self.env, self.cfg)

    def test_reset_returns_obs_and_infos(self):
        obs, infos = self.wrapped.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1"}

    def test_step_team_reward(self):
        self.wrapped.reset()
        actions = {a: 0 for a in self.wrapped.agents}
        obs, rewards, terms, truncs, infos = self.wrapped.step(actions)
        for a in ["agent_0", "agent_1"]:
            np.testing.assert_array_equal(infos[a]["vec_reward"], np.array([50.0, 60.0]))


class TestMOMASWrapperIndividualReward:
    def setup_method(self):
        self.env = StubEnv()
        self.cfg = MOMASConfig(
            reward_structure="individual",
            utility_type="individual",
            optimisation_criterion="SER",
            num_objectives=2,
            utility_functions={
                "agent_0": linear_utility(np.array([0.7, 0.3])),
                "agent_1": linear_utility(np.array([0.3, 0.7])),
            },
        )
        self.wrapped = MOMASWrapper(self.env, self.cfg)

    def test_step_individual_reward(self):
        self.wrapped.reset()
        actions = {a: 0 for a in self.wrapped.agents}
        obs, rewards, terms, truncs, infos = self.wrapped.step(actions)
        np.testing.assert_array_equal(infos["agent_0"]["vec_reward"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(infos["agent_1"]["vec_reward"], np.array([30.0, 40.0]))


from momas.utility import utilitarian_welfare

class TestMOMASConfigWelfare:
    def test_config_with_welfare_function(self):
        cfg = MOMASConfig(
            reward_structure="team",
            utility_type="social_choice",
            optimisation_criterion="SER",
            num_objectives=2,
            utility_functions={
                "agent_0": linear_utility(np.array([0.7, 0.3])),
                "agent_1": linear_utility(np.array([0.3, 0.7])),
            },
            welfare_function=utilitarian_welfare,
        )
        assert cfg.welfare_function is utilitarian_welfare

    def test_config_welfare_defaults_to_none(self):
        cfg = MOMASConfig(
            reward_structure="team",
            utility_type="team",
            optimisation_criterion="SER",
            num_objectives=2,
            utility_functions={"shared": linear_utility(np.array([0.5, 0.5]))},
        )
        assert cfg.welfare_function is None
