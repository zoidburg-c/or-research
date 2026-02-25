# tests/test_integration.py
import numpy as np
import pytest
from experiments.runner import ExperimentConfig, run_experiment


class TestIntegrationSmoke:
    """End-to-end test: environment + MOMAS wrapper + agent + training + evaluation."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path)

    @pytest.mark.parametrize("env_name,env_params", [
        ("trading", {"num_agents": 2, "agent_types": ["market_maker", "momentum"], "episode_length": 10}),
        ("healthcare", {"num_hospitals": 2, "episode_length": 10}),
    ])
    @pytest.mark.parametrize("reward_structure,utility_type", [
        ("team", "team"),
        ("team", "social_choice"),
        ("team", "individual"),
        ("individual", "social_choice"),
        ("individual", "individual"),
    ])
    def test_all_taxonomy_settings(
        self, env_name, env_params, reward_structure, utility_type, output_dir
    ):
        cfg = ExperimentConfig(
            env=env_name,
            env_params=env_params,
            momas_reward_structure=reward_structure,
            momas_utility_type=utility_type,
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=2,
            seed=42,
            output_dir=output_dir,
        )
        results = run_experiment(cfg)
        assert results["episode_returns"].shape == (2, 3)
        assert np.isfinite(results["hypervolume"])
        assert np.isfinite(results["ser"])
        assert np.isfinite(results["esr"])
