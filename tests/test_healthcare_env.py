# tests/test_healthcare_env.py
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from envs.healthcare.hospital_env import HealthcareEnv


class TestHealthcareEnvAPI:
    def test_pettingzoo_api_compliance(self):
        env = HealthcareEnv(
            num_hospitals=2,
            beds_per_hospital={"general": 10, "icu": 3, "emergency": 2},
            episode_length=20,
        )
        parallel_api_test(env, num_cycles=5)

    def test_reset_returns_correct_structure(self):
        env = HealthcareEnv(num_hospitals=2)
        obs, infos = env.reset(seed=42)
        assert len(obs) == 2
        assert all(env.observation_space(a).contains(obs[a]) for a in env.agents)

    def test_step_returns_vector_rewards(self):
        env = HealthcareEnv(num_hospitals=2)
        env.reset(seed=42)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in infos:
            assert "vec_reward" in infos[a]
            assert len(infos[a]["vec_reward"]) == 3
            assert "team_vec_reward" in infos[a]

    def test_episode_terminates(self):
        env = HealthcareEnv(num_hospitals=2, episode_length=5)
        env.reset(seed=42)
        for _ in range(10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        assert len(env.agents) == 0
