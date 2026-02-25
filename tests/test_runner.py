import pytest
import yaml
import numpy as np
from experiments.runner import ExperimentConfig, run_experiment, load_config


class TestExperimentConfig:
    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "env": "trading",
            "env_params": {
                "num_agents": 2,
                "agent_types": ["market_maker", "momentum"],
                "episode_length": 10,
            },
            "momas": {
                "reward_structure": "team",
                "utility_type": "team",
                "optimisation_criterion": "SER",
                "num_objectives": 3,
            },
            "agent_type": "tabular_moq",
            "num_episodes": 2,
            "seed": 42,
        }
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))
        cfg = load_config(str(config_file))
        assert cfg.env == "trading"
        assert cfg.seed == 42


class TestRunExperiment:
    def test_smoke_trading(self, tmp_path):
        cfg = ExperimentConfig(
            env="trading",
            env_params={
                "num_agents": 2,
                "agent_types": ["market_maker", "momentum"],
                "episode_length": 10,
            },
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=2,
            seed=42,
            output_dir=str(tmp_path),
        )
        results = run_experiment(cfg)
        assert "episode_returns" in results
        assert results["episode_returns"].shape[1] == 3

    def test_smoke_healthcare(self, tmp_path):
        cfg = ExperimentConfig(
            env="healthcare",
            env_params={
                "num_hospitals": 2,
                "episode_length": 10,
            },
            momas_reward_structure="individual",
            momas_utility_type="individual",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=2,
            seed=42,
            output_dir=str(tmp_path),
        )
        results = run_experiment(cfg)
        assert "episode_returns" in results
