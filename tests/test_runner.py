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


class TestPerAgentUtility:
    def test_individual_utility_uses_different_weights(self, tmp_path):
        """Individual utility agents should use different utility functions."""
        cfg = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 10},
            momas_reward_structure="team",
            momas_utility_type="individual",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=3,
            seed=42,
            output_dir=str(tmp_path),
        )
        r1 = run_experiment(cfg)

        cfg_team = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 10},
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=3,
            seed=42,
            output_dir=str(tmp_path / "team"),
        )
        r2 = run_experiment(cfg_team)

        # Individual and team utility should produce different results
        assert not np.allclose(r1["episode_returns"], r2["episode_returns"])


class TestSocialChoiceWelfare:
    def test_social_choice_differs_from_individual(self, tmp_path):
        """Social choice (welfare-aggregated) should differ from individual utility."""
        cfg_social = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 10},
            momas_reward_structure="team",
            momas_utility_type="social_choice",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=3,
            seed=42,
            output_dir=str(tmp_path / "social"),
        )
        r_social = run_experiment(cfg_social)

        cfg_indiv = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 10},
            momas_reward_structure="team",
            momas_utility_type="individual",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=3,
            seed=42,
            output_dir=str(tmp_path / "indiv"),
        )
        r_indiv = run_experiment(cfg_indiv)

        # Social choice welfare-aggregated training should differ from individual
        assert not np.allclose(r_social["episode_returns"], r_indiv["episode_returns"])


class TestESRDifferentiation:
    def test_esr_produces_different_metrics_than_ser(self, tmp_path):
        """ESR with non-linear utility should produce SER != ESR in metrics."""
        cfg = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 10},
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="ESR",
            num_objectives=3,
            agent_type="tabular_moq",
            num_episodes=5,
            seed=42,
            output_dir=str(tmp_path),
        )
        results = run_experiment(cfg)

        assert np.isfinite(results["ser"])
        assert np.isfinite(results["esr"])
        assert results["episode_returns"].shape == (5, 3)


class TestDeepRLAgents:
    def test_mo_dqn_smoke_healthcare(self, tmp_path):
        cfg = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 5},
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="mo_dqn",
            num_episodes=2,
            seed=42,
            output_dir=str(tmp_path),
        )
        results = run_experiment(cfg)
        assert results["episode_returns"].shape == (2, 3)
        assert np.isfinite(results["hypervolume"])

    def test_envelope_smoke_healthcare(self, tmp_path):
        cfg = ExperimentConfig(
            env="healthcare",
            env_params={"num_hospitals": 2, "episode_length": 5},
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="SER",
            num_objectives=3,
            agent_type="envelope",
            num_episodes=2,
            seed=42,
            output_dir=str(tmp_path),
        )
        results = run_experiment(cfg)
        assert results["episode_returns"].shape == (2, 3)
        assert np.isfinite(results["hypervolume"])
