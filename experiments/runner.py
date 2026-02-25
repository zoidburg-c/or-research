from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from momas.base_env import MOMASConfig, MOMASWrapper
from momas.utility import linear_utility
from momas.metrics import compute_pareto_front, hypervolume, evaluate_ser, evaluate_esr
from agents.mo_q_learning import TabularMOQLearning


@dataclass
class ExperimentConfig:
    env: str
    env_params: dict
    momas_reward_structure: str
    momas_utility_type: str
    momas_criterion: str
    num_objectives: int
    agent_type: str
    num_episodes: int
    seed: int
    output_dir: str = "results"


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    momas = data.get("momas", {})
    return ExperimentConfig(
        env=data["env"],
        env_params=data.get("env_params", {}),
        momas_reward_structure=momas.get("reward_structure", "team"),
        momas_utility_type=momas.get("utility_type", "team"),
        momas_criterion=momas.get("optimisation_criterion", "SER"),
        num_objectives=momas.get("num_objectives", 3),
        agent_type=data.get("agent_type", "tabular_moq"),
        num_episodes=data.get("num_episodes", 100),
        seed=data.get("seed", 0),
        output_dir=data.get("output_dir", "results"),
    )


def _make_env(cfg: ExperimentConfig):
    if cfg.env == "trading":
        from envs.trading.orderbook_env import TradingEnv
        return TradingEnv(**cfg.env_params)
    elif cfg.env == "healthcare":
        from envs.healthcare.hospital_env import HealthcareEnv
        return HealthcareEnv(**cfg.env_params)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


def _make_momas_config(cfg: ExperimentConfig, agents: list[str]) -> MOMASConfig:
    weights = np.ones(cfg.num_objectives) / cfg.num_objectives

    if cfg.momas_utility_type == "team":
        utility_fns = {"shared": linear_utility(weights)}
    else:
        rng = np.random.default_rng(cfg.seed)
        utility_fns = {}
        for a in agents:
            w = rng.dirichlet(np.ones(cfg.num_objectives))
            utility_fns[a] = linear_utility(w)

    return MOMASConfig(
        reward_structure=cfg.momas_reward_structure,
        utility_type=cfg.momas_utility_type,
        optimisation_criterion=cfg.momas_criterion,
        num_objectives=cfg.num_objectives,
        utility_functions=utility_fns,
    )


def _discretize_obs(obs: np.ndarray, num_bins: int = 10) -> int:
    binned = np.clip(np.digitize(obs, np.linspace(obs.min() - 1, obs.max() + 1, num_bins)), 0, num_bins - 1)
    h = 0
    for b in binned[:5]:
        h = h * num_bins + int(b)
    return h % 10000


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    base_env = _make_env(cfg)
    base_env.reset(seed=cfg.seed)
    momas_cfg = _make_momas_config(cfg, base_env.possible_agents)
    env = MOMASWrapper(base_env, momas_cfg)

    weights = np.ones(cfg.num_objectives) / cfg.num_objectives
    agents = {}
    for a in env.possible_agents:
        agents[a] = TabularMOQLearning(
            num_states=10000,
            num_actions=env.action_space(a).n,
            num_objectives=cfg.num_objectives,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=0.3,
            seed=rng.integers(0, 2**31),
        )

    all_episode_returns = []

    for ep in range(cfg.num_episodes):
        obs, infos = env.reset(seed=int(rng.integers(0, 2**31)))
        episode_rewards = {a: np.zeros(cfg.num_objectives) for a in env.possible_agents}

        while env.agents:
            actions = {}
            states = {}
            for a in env.agents:
                s = _discretize_obs(obs[a])
                states[a] = s
                actions[a] = agents[a].select_action(s, weights)

            obs, rewards, terms, truncs, infos = env.step(actions)

            for a in states:
                if a in infos:
                    vec_r = infos[a]["vec_reward"]
                    episode_rewards[a] += vec_r
                    next_s = _discretize_obs(obs[a]) if a in obs else states[a]
                    agents[a].update(states[a], actions[a], vec_r, next_s, weights)

        mean_return = np.mean(list(episode_rewards.values()), axis=0)
        all_episode_returns.append(mean_return)

    episode_returns = np.array(all_episode_returns)

    shared_u = linear_utility(weights)
    ser = evaluate_ser(episode_returns, shared_u)
    esr = evaluate_esr(episode_returns, shared_u)
    pareto_front = compute_pareto_front(episode_returns)
    ref_point = np.min(episode_returns, axis=0) - 1.0
    hv = hypervolume(pareto_front, ref_point)

    results = {
        "episode_returns": episode_returns,
        "pareto_front": pareto_front,
        "hypervolume": hv,
        "ser": ser,
        "esr": esr,
        "config": cfg,
    }

    out_dir = Path(cfg.output_dir) / cfg.env / f"{cfg.momas_reward_structure}-{cfg.momas_utility_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"seed_{cfg.seed}.npz",
        episode_returns=episode_returns,
        pareto_front=pareto_front,
        hypervolume=np.array([hv]),
        ser=np.array([ser]),
        esr=np.array([esr]),
    )

    return results
