from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import functools
import numpy as np
from collections.abc import Callable
from pettingzoo import ParallelEnv


@dataclass
class MOMASConfig:
    """Configuration for a MOMAS taxonomy setting."""

    reward_structure: Literal["team", "individual"]
    utility_type: Literal["team", "social_choice", "individual"]
    optimisation_criterion: Literal["SER", "ESR"]
    num_objectives: int
    utility_functions: dict[str, Callable]
    welfare_function: Callable | None = None

    def __post_init__(self):
        if self.reward_structure == "individual" and self.utility_type == "team":
            raise ValueError(
                "Individual rewards with team utility is not a meaningful combination — "
                "even identical utility functions produce different scalar values "
                "from different reward vectors."
            )


class MOMASWrapper(ParallelEnv):
    """Wraps a PettingZoo ParallelEnv with MOMAS taxonomy configuration.

    The inner environment must provide vector rewards in infos:
    - infos[agent]["vec_reward"]: per-agent reward vector (for individual reward)
    - infos[agent]["team_vec_reward"]: shared team reward vector (for team reward)
    """

    metadata = {"name": "momas_wrapper_v0"}

    def __init__(self, env: ParallelEnv, config: MOMASConfig):
        self._env = env
        self.config = config
        self.possible_agents = env.possible_agents
        self.agents = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._env.action_space(agent)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        obs, infos = self._env.reset(seed=seed, options=options)
        self.agents = self._env.agents[:]
        return obs, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict, dict, dict, dict, dict]:
        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        self.agents = self._env.agents[:]

        if self.config.reward_structure == "team":
            any_agent = next(iter(infos))
            team_reward = infos[any_agent]["team_vec_reward"]
            for agent in infos:
                infos[agent]["vec_reward"] = np.array(team_reward, dtype=np.float64)

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
