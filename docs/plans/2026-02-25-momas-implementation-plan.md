# MOMAS Research Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build working multi-objective multi-agent RL simulations for algorithmic trading and healthcare resource allocation, implementing all 5 taxonomy settings from Radulescu et al. (2019).

**Architecture:** Monolithic Python package with a shared MOMAS framework layer that wraps PettingZoo `ParallelEnv` environments. The framework handles taxonomy configuration (reward structure, utility type, SER/ESR criterion) so domain environments only define state/action/transition logic. MORL-Baselines agents are wrapped for multi-agent use. A config-driven experiment runner executes the full 5x2 taxonomy sweep.

**Tech Stack:** Python 3.11+, PettingZoo, MO-Gymnasium, MORL-Baselines, PyTorch, NumPy, pymoo, pytest, PyYAML, Matplotlib/Seaborn

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `momas/__init__.py`
- Create: `momas/utility.py` (empty)
- Create: `momas/metrics.py` (empty)
- Create: `momas/base_env.py` (empty)
- Create: `envs/__init__.py`
- Create: `envs/trading/__init__.py`
- Create: `envs/healthcare/__init__.py`
- Create: `agents/__init__.py`
- Create: `experiments/__init__.py`
- Create: `experiments/configs/.gitkeep`
- Create: `results/.gitignore`
- Create: `notebooks/.gitkeep`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "momas-research"
version = "0.1.0"
description = "Multi-Objective Multi-Agent Decision Making research prototype"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "torch>=2.1",
    "pettingzoo>=1.25",
    "mo-gymnasium>=0.3",
    "morl-baselines>=1.1",
    "gymnasium>=1.0",
    "pymoo>=0.6",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create all directory `__init__.py` files and placeholders**

Create every `__init__.py` and placeholder file listed above. All `__init__.py` files are empty. `results/.gitignore` contains `*\n!.gitignore`. `.gitignore` contains standard Python ignores plus `results/`.

**Step 3: Install in editable mode**

Run: `pip install -e ".[dev]"`
Expected: Successful install with all dependencies resolved.

**Step 4: Verify imports work**

Run: `python -c "import momas; import envs; import agents; import experiments; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Utility Functions

**Files:**
- Create: `tests/test_utility.py`
- Modify: `momas/utility.py`

**Step 1: Write failing tests**

```python
# tests/test_utility.py
import numpy as np
import pytest
from momas.utility import (
    linear_utility,
    threshold_utility,
    chebyshev_utility,
    utilitarian_welfare,
    nash_welfare,
    lorenz_welfare,
)


class TestLinearUtility:
    def test_weighted_sum(self):
        u = linear_utility(np.array([0.5, 0.3, 0.2]))
        result = u(np.array([10.0, 20.0, 30.0]))
        assert result == pytest.approx(17.0)  # 5 + 6 + 6

    def test_uniform_weights(self):
        u = linear_utility(np.array([1.0, 1.0]))
        assert u(np.array([3.0, 4.0])) == pytest.approx(7.0)

    def test_zero_reward(self):
        u = linear_utility(np.array([0.5, 0.5]))
        assert u(np.array([0.0, 0.0])) == pytest.approx(0.0)


class TestThresholdUtility:
    def test_above_thresholds(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        # surplus: [5.0, 7.0], weighted sum = 12.0
        result = u(np.array([10.0, 10.0]))
        assert result == pytest.approx(12.0)

    def test_below_threshold_returns_negative_infinity(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        result = u(np.array([4.0, 10.0]))
        assert result == float("-inf")

    def test_exactly_at_threshold(self):
        u = threshold_utility(
            thresholds=np.array([5.0, 3.0]),
            weights=np.array([1.0, 1.0]),
        )
        result = u(np.array([5.0, 3.0]))
        assert result == pytest.approx(0.0)


class TestChebyshevUtility:
    def test_balanced_solution(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        # max weighted deviation from ideal: max(1*5, 1*5) = 5, negated = -5
        result = u(np.array([5.0, 5.0]))
        assert result == pytest.approx(-5.0)

    def test_unbalanced_penalized(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        # max(1*1, 1*9) = 9, negated = -9
        result = u(np.array([9.0, 1.0]))
        assert result == pytest.approx(-9.0)

    def test_at_ideal(self):
        u = chebyshev_utility(
            weights=np.array([1.0, 1.0]),
            ideal=np.array([10.0, 10.0]),
        )
        assert u(np.array([10.0, 10.0])) == pytest.approx(0.0)


class TestWelfareFunctions:
    def test_utilitarian(self):
        assert utilitarian_welfare([3.0, 4.0, 5.0]) == pytest.approx(12.0)

    def test_nash(self):
        assert nash_welfare([2.0, 3.0, 4.0]) == pytest.approx(24.0)

    def test_nash_with_zero(self):
        assert nash_welfare([0.0, 3.0]) == pytest.approx(0.0)

    def test_lorenz_prefers_equality(self):
        equal = lorenz_welfare([5.0, 5.0, 5.0])
        unequal = lorenz_welfare([1.0, 5.0, 9.0])
        assert equal > unequal

    def test_lorenz_same_sum(self):
        # Same total utility but different distribution
        equal = lorenz_welfare([5.0, 5.0])
        unequal = lorenz_welfare([2.0, 8.0])
        assert equal > unequal
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_utility.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement utility functions**

```python
# momas/utility.py
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def linear_utility(weights: np.ndarray) -> Callable[[np.ndarray], float]:
    """Weighted sum: u(r) = w . r"""
    w = np.asarray(weights, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        return float(np.dot(w, rewards))

    return _u


def threshold_utility(
    thresholds: np.ndarray,
    weights: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Returns -inf if any objective is below its threshold,
    otherwise returns the weighted sum of the surplus above thresholds."""
    t = np.asarray(thresholds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        r = np.asarray(rewards, dtype=np.float64)
        if np.any(r < t):
            return float("-inf")
        return float(np.dot(w, r - t))

    return _u


def chebyshev_utility(
    weights: np.ndarray,
    ideal: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Negated weighted Chebyshev distance from ideal point.
    u(r) = -max_d( w_d * |ideal_d - r_d| )
    Higher (less negative) is better. 0 at ideal point."""
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(ideal, dtype=np.float64)

    def _u(rewards: np.ndarray) -> float:
        r = np.asarray(rewards, dtype=np.float64)
        return float(-np.max(w * np.abs(z - r)))

    return _u


def utilitarian_welfare(utilities: list[float]) -> float:
    """Sum of individual utilities."""
    return sum(utilities)


def nash_welfare(utilities: list[float]) -> float:
    """Product of individual utilities."""
    return float(math.prod(utilities))


def lorenz_welfare(utilities: list[float]) -> float:
    """Sum of cumulative sorted utilities (Lorenz-based).
    Prefers more equal distributions: for same total,
    equal distributions produce higher Lorenz welfare."""
    sorted_u = sorted(utilities)
    cumulative = 0.0
    total = 0.0
    for u in sorted_u:
        cumulative += u
        total += cumulative
    return total
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utility.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add momas/utility.py tests/test_utility.py
git commit -m "feat: add utility functions (linear, threshold, chebyshev) and welfare aggregators"
```

---

## Task 3: Evaluation Metrics

**Files:**
- Create: `tests/test_metrics.py`
- Modify: `momas/metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from momas.metrics import compute_pareto_front, hypervolume, evaluate_ser, evaluate_esr
from momas.utility import linear_utility


class TestParetoFront:
    def test_simple_2d(self):
        # Points: (1,5), (2,3), (3,2), (4,1), (3,4)
        # Pareto front: (1,5), (2,3), (3,2), (4,1) — (3,4) dominated by (1,5)? No.
        # Actually: (1,5) dominates (3,4)? 1<3 on obj0 but 5>4 on obj1. Not dominated.
        # Wait — for maximization: (3,4) is dominated by nothing that's >= on all.
        # (1,5): obj0=1, obj1=5. (3,4): obj0=3, obj1=4. Neither dominates the other.
        # Pareto front (maximizing both): (1,5), (3,4), (4,1) — no, (2,3) vs (3,4):
        # 3>2 and 4>3, so (3,4) dominates (2,3). (3,2) dominated by (3,4)? 3>=3, 4>2 yes.
        # Front: (1,5), (3,4), (4,1)
        points = np.array([[1, 5], [2, 3], [3, 2], [4, 1], [3, 4]])
        front = compute_pareto_front(points)
        assert len(front) == 3
        expected = {(1, 5), (3, 4), (4, 1)}
        actual = {tuple(p) for p in front}
        assert actual == expected

    def test_single_point(self):
        points = np.array([[1.0, 2.0]])
        front = compute_pareto_front(points)
        assert len(front) == 1

    def test_all_same(self):
        points = np.array([[1.0, 1.0], [1.0, 1.0]])
        front = compute_pareto_front(points)
        assert len(front) == 1


class TestHypervolume:
    def test_simple_2d(self):
        # Two points forming a known area relative to ref_point
        front = np.array([[3.0, 1.0], [1.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        # Area: 1*3 + (3-1)*1 = 3 + 2 = 5
        assert hv == pytest.approx(5.0)

    def test_single_point(self):
        front = np.array([[2.0, 3.0]])
        ref = np.array([0.0, 0.0])
        hv = hypervolume(front, ref)
        assert hv == pytest.approx(6.0)


class TestSERESR:
    def test_ser_linear(self):
        # SER: u(E[returns])
        # Episode returns: [10, 20], [30, 40] -> mean [20, 30]
        # u([20,30]) with weights [0.5, 0.5] = 25.0
        episode_returns = np.array([[10.0, 20.0], [30.0, 40.0]])
        u = linear_utility(np.array([0.5, 0.5]))
        ser = evaluate_ser(episode_returns, u)
        assert ser == pytest.approx(25.0)

    def test_esr_linear(self):
        # ESR: E[u(returns)]
        # u([10,20]) = 15.0, u([30,40]) = 35.0 -> mean = 25.0
        episode_returns = np.array([[10.0, 20.0], [30.0, 40.0]])
        u = linear_utility(np.array([0.5, 0.5]))
        esr = evaluate_esr(episode_returns, u)
        assert esr == pytest.approx(25.0)

    def test_ser_esr_differ_nonlinear(self):
        # For non-linear utility, SER != ESR
        episode_returns = np.array([[1.0, 9.0], [9.0, 1.0]])
        # Use a non-linear utility: product of objectives
        def product_u(r):
            return float(r[0] * r[1])
        ser = evaluate_ser(episode_returns, product_u)  # u(mean([1,9],[9,1])) = u([5,5]) = 25
        esr = evaluate_esr(episode_returns, product_u)  # mean(u([1,9]), u([9,1])) = mean(9,9) = 9
        assert ser == pytest.approx(25.0)
        assert esr == pytest.approx(9.0)
        assert ser != pytest.approx(esr)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement metrics**

```python
# momas/metrics.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """Compute the Pareto front from a set of points (maximization convention).

    Args:
        points: shape (N, d) — N solution vectors with d objectives.

    Returns:
        shape (K, d) — the K non-dominated points.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points.copy()

    # pymoo uses minimization — negate for maximization
    nds = NonDominatedSorting()
    front_idx = nds.do(-points, only_non_dominated_front=True)

    # Deduplicate
    front = points[front_idx]
    unique = np.unique(front, axis=0)
    return unique


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume indicator (maximization convention).

    Args:
        front: shape (K, d) — Pareto front points.
        ref_point: shape (d,) — reference point (must be dominated by all front points).

    Returns:
        Hypervolume scalar value.
    """
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)

    # pymoo uses minimization — negate both
    indicator = HV(ref_point=-ref)
    return float(indicator(-front))


def evaluate_ser(
    episode_returns: np.ndarray,
    utility_fn: Callable[[np.ndarray], float],
) -> float:
    """Scalarised Expected Returns: apply utility to expected return vector.
    V_SER = u(E[returns])

    Args:
        episode_returns: shape (num_episodes, d) — per-episode return vectors.
        utility_fn: maps a return vector to a scalar.
    """
    expected = np.mean(episode_returns, axis=0)
    return utility_fn(expected)


def evaluate_esr(
    episode_returns: np.ndarray,
    utility_fn: Callable[[np.ndarray], float],
) -> float:
    """Expected Scalarised Returns: average utility over individual episodes.
    V_ESR = E[u(returns)]

    Args:
        episode_returns: shape (num_episodes, d) — per-episode return vectors.
        utility_fn: maps a return vector to a scalar.
    """
    utilities = [utility_fn(r) for r in episode_returns]
    return float(np.mean(utilities))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add momas/metrics.py tests/test_metrics.py
git commit -m "feat: add Pareto front, hypervolume, SER/ESR evaluation metrics"
```

---

## Task 4: MOMAS Environment Wrapper

**Files:**
- Create: `tests/test_base_env.py`
- Modify: `momas/base_env.py`

**Step 1: Write failing tests**

```python
# tests/test_base_env.py
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
        # Individual reward vectors per agent
        self._rewards = {
            "agent_0": np.array([10.0, 20.0]),
            "agent_1": np.array([30.0, 40.0]),
        }
        # Team (global) reward
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
        rewards = {a: 0.0 for a in self.agents}  # scalar placeholder
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
        # Individual reward + team utility is not meaningful per the paper
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
        # Team reward: both agents get team_vec_reward [50, 60]
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
        # Individual: each agent keeps own vec_reward
        np.testing.assert_array_equal(infos["agent_0"]["vec_reward"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(infos["agent_1"]["vec_reward"], np.array([30.0, 40.0]))
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_base_env.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement MOMAS wrapper**

```python
# momas/base_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import functools
import numpy as np
from collections.abc import Callable
from pettingzoo import ParallelEnv


@dataclass
class MOMASConfig:
    """Configuration for a MOMAS taxonomy setting.

    Args:
        reward_structure: "team" (all agents get same reward) or "individual".
        utility_type: "team", "social_choice", or "individual".
        optimisation_criterion: "SER" or "ESR".
        num_objectives: number of objective dimensions.
        utility_functions: dict mapping agent name (or "shared") to a utility callable.
    """

    reward_structure: Literal["team", "individual"]
    utility_type: Literal["team", "social_choice", "individual"]
    optimisation_criterion: Literal["SER", "ESR"]
    num_objectives: int
    utility_functions: dict[str, Callable]

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

        # Apply reward structure transformation
        if self.config.reward_structure == "team":
            # All agents get the team reward vector
            # Use team_vec_reward from any active agent's info
            any_agent = next(iter(infos))
            team_reward = infos[any_agent]["team_vec_reward"]
            for agent in infos:
                infos[agent]["vec_reward"] = np.array(team_reward, dtype=np.float64)
        # For "individual", leave vec_reward as-is from the inner env

        return obs, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_base_env.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add momas/base_env.py tests/test_base_env.py
git commit -m "feat: add MOMASConfig and MOMASWrapper for taxonomy-configurable environments"
```

---

## Task 5: Order Book Matching Engine

**Files:**
- Create: `tests/test_orderbook.py`
- Create: `envs/trading/orderbook.py`

**Step 1: Write failing tests**

```python
# tests/test_orderbook.py
import pytest
from envs.trading.orderbook import LimitOrderBook, Order, Side


class TestOrderBook:
    def setup_method(self):
        self.book = LimitOrderBook(tick_size=0.01)

    def test_add_limit_buy(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=100.0, quantity=10))
        assert self.book.best_bid() == 100.0
        assert self.book.best_ask() is None

    def test_add_limit_sell(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.best_ask() == 101.0
        assert self.book.best_bid() is None

    def test_crossing_order_executes(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.BUY, price=100.0, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0
        assert trades[0].quantity == 5
        assert trades[0].buyer == "b"
        assert trades[0].seller == "a"
        # Remaining sell quantity
        assert self.book.best_ask() == 100.0

    def test_price_time_priority(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=5))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=100.0, quantity=5))
        trades = self.book.add_order(Order(agent="c", side=Side.BUY, price=100.0, quantity=5))
        assert len(trades) == 1
        assert trades[0].seller == "a"  # first in gets filled

    def test_market_buy(self):
        self.book.add_order(Order(agent="a", side=Side.SELL, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.BUY, price=None, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0

    def test_market_sell(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=100.0, quantity=10))
        trades = self.book.add_order(Order(agent="b", side=Side.SELL, price=None, quantity=5))
        assert len(trades) == 1
        assert trades[0].price == 100.0

    def test_cancel_order(self):
        order = Order(agent="a", side=Side.BUY, price=100.0, quantity=10)
        self.book.add_order(order)
        assert self.book.best_bid() == 100.0
        self.book.cancel_order(order.order_id)
        assert self.book.best_bid() is None

    def test_mid_price(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.mid_price() == pytest.approx(100.0)

    def test_spread(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=10))
        assert self.book.spread() == pytest.approx(2.0)

    def test_depth(self):
        self.book.add_order(Order(agent="a", side=Side.BUY, price=99.0, quantity=10))
        self.book.add_order(Order(agent="a", side=Side.BUY, price=98.0, quantity=5))
        self.book.add_order(Order(agent="b", side=Side.SELL, price=101.0, quantity=7))
        bids, asks = self.book.depth(levels=5)
        assert len(bids) == 2
        assert bids[0] == (99.0, 10)  # best bid first
        assert len(asks) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orderbook.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement the order book**

```python
# envs/trading/orderbook.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple
import itertools


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


_order_counter = itertools.count()


@dataclass
class Order:
    agent: str
    side: Side
    price: float | None  # None = market order
    quantity: int
    order_id: int = field(default_factory=lambda: next(_order_counter))
    timestamp: int = 0


class Trade(NamedTuple):
    buyer: str
    seller: str
    price: float
    quantity: int


class LimitOrderBook:
    """Simplified limit order book with price-time priority."""

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        # bids: price -> list of orders (highest price = best)
        self._bids: dict[float, list[Order]] = defaultdict(list)
        # asks: price -> list of orders (lowest price = best)
        self._asks: dict[float, list[Order]] = defaultdict(list)
        self._orders: dict[int, Order] = {}
        self._timestamp = 0

    def add_order(self, order: Order) -> list[Trade]:
        """Add an order to the book, executing any trades that result."""
        self._timestamp += 1
        order.timestamp = self._timestamp
        trades: list[Trade] = []

        if order.side == Side.BUY:
            trades = self._match_buy(order)
        else:
            trades = self._match_sell(order)

        # If order has remaining quantity and is a limit order, rest in book
        if order.quantity > 0 and order.price is not None:
            book = self._bids if order.side == Side.BUY else self._asks
            book[order.price].append(order)
            self._orders[order.order_id] = order

        return trades

    def _match_buy(self, order: Order) -> list[Trade]:
        trades = []
        while order.quantity > 0 and self._asks:
            best_ask_price = min(self._asks.keys())
            if order.price is not None and order.price < best_ask_price:
                break  # buy price too low
            trades.extend(self._fill_against(order, self._asks, best_ask_price, Side.BUY))
        return trades

    def _match_sell(self, order: Order) -> list[Trade]:
        trades = []
        while order.quantity > 0 and self._bids:
            best_bid_price = max(self._bids.keys())
            if order.price is not None and order.price > best_bid_price:
                break  # sell price too high
            trades.extend(self._fill_against(order, self._bids, best_bid_price, Side.SELL))
        return trades

    def _fill_against(
        self,
        aggressor: Order,
        book_side: dict[float, list[Order]],
        price_level: float,
        aggressor_side: Side,
    ) -> list[Trade]:
        trades = []
        queue = book_side[price_level]
        while aggressor.quantity > 0 and queue:
            resting = queue[0]
            fill_qty = min(aggressor.quantity, resting.quantity)

            buyer = aggressor.agent if aggressor_side == Side.BUY else resting.agent
            seller = resting.agent if aggressor_side == Side.BUY else aggressor.agent

            trades.append(Trade(buyer=buyer, seller=seller, price=price_level, quantity=fill_qty))

            aggressor.quantity -= fill_qty
            resting.quantity -= fill_qty

            if resting.quantity == 0:
                queue.pop(0)
                self._orders.pop(resting.order_id, None)

        if not queue:
            del book_side[price_level]

        return trades

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order by ID. Returns True if found and cancelled."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._bids if order.side == Side.BUY else self._asks
        if order.price in book:
            queue = book[order.price]
            queue[:] = [o for o in queue if o.order_id != order_id]
            if not queue:
                del book[order.price]
        return True

    def best_bid(self) -> float | None:
        return max(self._bids.keys()) if self._bids else None

    def best_ask(self) -> float | None:
        return min(self._asks.keys()) if self._asks else None

    def mid_price(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, levels: int = 5) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """Return top N price levels for bids and asks.
        Bids: highest first. Asks: lowest first.
        Each entry: (price, total_quantity)."""
        bids = sorted(self._bids.keys(), reverse=True)[:levels]
        asks = sorted(self._asks.keys())[:levels]
        bid_depth = [(p, sum(o.quantity for o in self._bids[p])) for p in bids]
        ask_depth = [(p, sum(o.quantity for o in self._asks[p])) for p in asks]
        return bid_depth, ask_depth

    def get_agent_orders(self, agent: str) -> list[Order]:
        """Get all active orders for an agent."""
        return [o for o in self._orders.values() if o.agent == agent]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orderbook.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add envs/trading/orderbook.py tests/test_orderbook.py
git commit -m "feat: add limit order book matching engine with price-time priority"
```

---

## Task 6: Trading PettingZoo Environment

**Files:**
- Create: `tests/test_trading_env.py`
- Modify: `envs/trading/orderbook_env.py`

**Step 1: Write failing tests**

```python
# tests/test_trading_env.py
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from envs.trading.orderbook_env import TradingEnv


class TestTradingEnvAPI:
    def test_pettingzoo_api_compliance(self):
        env = TradingEnv(
            num_agents=2,
            agent_types=["market_maker", "momentum"],
            num_objectives=3,
            episode_length=20,
        )
        parallel_api_test(env, num_cycles=5)

    def test_reset_returns_correct_structure(self):
        env = TradingEnv(num_agents=2, agent_types=["market_maker", "momentum"])
        obs, infos = env.reset(seed=42)
        assert len(obs) == 2
        assert all(env.observation_space(a).contains(obs[a]) for a in env.agents)

    def test_step_returns_vector_rewards(self):
        env = TradingEnv(num_agents=2, agent_types=["market_maker", "momentum"])
        env.reset(seed=42)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a in infos:
            assert "vec_reward" in infos[a]
            assert len(infos[a]["vec_reward"]) == 3  # P&L, risk, liquidity cost
            assert "team_vec_reward" in infos[a]

    def test_episode_terminates(self):
        env = TradingEnv(num_agents=2, agent_types=["market_maker", "momentum"], episode_length=5)
        env.reset(seed=42)
        for _ in range(10):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        assert len(env.agents) == 0

    def test_agent_types_length_matches_num_agents(self):
        with pytest.raises(ValueError):
            TradingEnv(num_agents=3, agent_types=["market_maker", "momentum"])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_env.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement trading environment**

```python
# envs/trading/orderbook_env.py
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from envs.trading.orderbook import LimitOrderBook, Order, Side


# Action encoding:
# 0: hold
# 1-5: limit buy at mid - (1..5) ticks
# 6-10: limit sell at mid + (1..5) ticks
# 11: market buy
# 12: market sell
# 13: cancel all own orders
NUM_ACTIONS = 14
ORDER_QUANTITY = 10  # fixed order size for simplicity


class TradingEnv(ParallelEnv):
    """Multi-agent order book trading environment.

    Each agent interacts with a shared limit order book.
    Rewards are 3-objective vectors: [P&L, risk, liquidity_cost].
    """

    metadata = {"name": "trading_v0"}

    def __init__(
        self,
        num_agents: int = 4,
        agent_types: list[str] | None = None,
        num_objectives: int = 3,
        orderbook_depth: int = 10,
        tick_size: float = 0.01,
        initial_cash: float = 100_000.0,
        initial_price: float = 100.0,
        episode_length: int = 1000,
        price_history_len: int = 50,
    ):
        if agent_types is None:
            agent_types = ["market_maker"] * num_agents
        if len(agent_types) != num_agents:
            raise ValueError(
                f"agent_types length ({len(agent_types)}) must match num_agents ({num_agents})"
            )

        self.num_objectives = num_objectives
        self.orderbook_depth = orderbook_depth
        self.tick_size = tick_size
        self.initial_cash = initial_cash
        self.initial_price = initial_price
        self.episode_length = episode_length
        self.price_history_len = price_history_len

        self.possible_agents = [f"trader_{i}" for i in range(num_agents)]
        self.agent_types = {self.possible_agents[i]: agent_types[i] for i in range(num_agents)}
        self.agents = []

        # State tracked per agent
        self._cash: dict[str, float] = {}
        self._inventory: dict[str, int] = {}
        self._peak_value: dict[str, float] = {}

        self._book: LimitOrderBook | None = None
        self._price_history: list[float] = []
        self._step_count = 0
        self._rng: np.random.Generator | None = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        # [mid_price, spread, cash, inventory, unrealized_pnl,
        #  price_history(50), bid_depth(10*2), ask_depth(10*2)]
        size = 5 + self.price_history_len + self.orderbook_depth * 4
        return Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        return Discrete(NUM_ACTIONS)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self._rng = np.random.default_rng(seed)
        self._book = LimitOrderBook(tick_size=self.tick_size)
        self._step_count = 0
        self._price_history = [self.initial_price] * self.price_history_len
        self.agents = self.possible_agents[:]

        for a in self.agents:
            self._cash[a] = self.initial_cash
            self._inventory[a] = 0
            self._peak_value[a] = self.initial_cash

        # Seed the book with initial liquidity
        for offset in range(1, 6):
            self._book.add_order(
                Order(agent="__init__", side=Side.BUY,
                      price=self.initial_price - offset * self.tick_size, quantity=100)
            )
            self._book.add_order(
                Order(agent="__init__", side=Side.SELL,
                      price=self.initial_price + offset * self.tick_size, quantity=100)
            )

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._get_info(a, np.zeros(self.num_objectives)) for a in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict, dict, dict, dict]:
        self._step_count += 1
        agent_trades: dict[str, list] = {a: [] for a in self.agents}

        # Process actions for each agent
        for agent, action in actions.items():
            trades = self._process_action(agent, action)
            # Record trades for reward computation
            for t in trades:
                if t.buyer == agent:
                    agent_trades[agent].append(t)
                if t.seller == agent:
                    agent_trades[agent].append(t)

        # Update price history
        mid = self._book.mid_price()
        if mid is not None:
            self._price_history.append(mid)
        else:
            self._price_history.append(self._price_history[-1])
        if len(self._price_history) > self.price_history_len:
            self._price_history = self._price_history[-self.price_history_len:]

        # Compute rewards
        vec_rewards = {}
        for a in self.agents:
            vec_rewards[a] = self._compute_reward(a, agent_trades[a])

        # Team reward = sum of all individual rewards
        team_reward = np.sum(list(vec_rewards.values()), axis=0)

        done = self._step_count >= self.episode_length
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: self._get_info(a, vec_rewards[a], team_reward) for a in self.agents}

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _process_action(self, agent: str, action: int) -> list:
        mid = self._book.mid_price() or self.initial_price
        trades = []

        if action == 0:  # hold
            pass
        elif 1 <= action <= 5:  # limit buy at mid - offset
            offset = action
            price = round(mid - offset * self.tick_size, 8)
            trades = self._book.add_order(
                Order(agent=agent, side=Side.BUY, price=price, quantity=ORDER_QUANTITY)
            )
        elif 6 <= action <= 10:  # limit sell at mid + offset
            offset = action - 5
            price = round(mid + offset * self.tick_size, 8)
            trades = self._book.add_order(
                Order(agent=agent, side=Side.SELL, price=price, quantity=ORDER_QUANTITY)
            )
        elif action == 11:  # market buy
            trades = self._book.add_order(
                Order(agent=agent, side=Side.BUY, price=None, quantity=ORDER_QUANTITY)
            )
        elif action == 12:  # market sell
            trades = self._book.add_order(
                Order(agent=agent, side=Side.SELL, price=None, quantity=ORDER_QUANTITY)
            )
        elif action == 13:  # cancel all
            for order in self._book.get_agent_orders(agent):
                self._book.cancel_order(order.order_id)

        # Update cash and inventory from trades
        for t in trades:
            if t.buyer == agent:
                self._cash[agent] -= t.price * t.quantity
                self._inventory[agent] += t.quantity
            if t.seller == agent:
                self._cash[agent] += t.price * t.quantity
                self._inventory[agent] -= t.quantity

        return trades

    def _compute_reward(self, agent: str, trades: list) -> np.ndarray:
        mid = self._book.mid_price() or self._price_history[-1]

        # Objective 1: P&L (change in portfolio value this step)
        portfolio_value = self._cash[agent] + self._inventory[agent] * mid
        pnl = portfolio_value - self.initial_cash

        # Objective 2: Risk (negative drawdown — higher is better)
        if portfolio_value > self._peak_value[agent]:
            self._peak_value[agent] = portfolio_value
        drawdown = (self._peak_value[agent] - portfolio_value) / max(self._peak_value[agent], 1.0)
        risk = -drawdown  # 0 = no drawdown, negative = drawdown

        # Objective 3: Liquidity cost (negative slippage — higher is better)
        slippage = 0.0
        for t in trades:
            slippage += abs(t.price - mid) * t.quantity
        liquidity_cost = -slippage

        return np.array([pnl, risk, liquidity_cost], dtype=np.float64)

    def _get_obs(self, agent: str) -> np.ndarray:
        mid = self._book.mid_price() or self._price_history[-1]
        spread = self._book.spread() or 0.0
        unrealized = self._inventory[agent] * mid

        # Price history (padded)
        ph = np.array(self._price_history[-self.price_history_len:], dtype=np.float32)
        if len(ph) < self.price_history_len:
            ph = np.pad(ph, (self.price_history_len - len(ph), 0), constant_values=mid)

        # Order book depth
        bids, asks = self._book.depth(levels=self.orderbook_depth)
        bid_flat = np.zeros(self.orderbook_depth * 2, dtype=np.float32)
        ask_flat = np.zeros(self.orderbook_depth * 2, dtype=np.float32)
        for i, (p, q) in enumerate(bids):
            bid_flat[i * 2] = p
            bid_flat[i * 2 + 1] = q
        for i, (p, q) in enumerate(asks):
            ask_flat[i * 2] = p
            ask_flat[i * 2 + 1] = q

        obs = np.concatenate([
            np.array([mid, spread, self._cash[agent], self._inventory[agent], unrealized],
                     dtype=np.float32),
            ph,
            bid_flat,
            ask_flat,
        ])
        return obs

    def _get_info(
        self, agent: str, vec_reward: np.ndarray, team_reward: np.ndarray | None = None
    ) -> dict:
        info = {"vec_reward": vec_reward}
        if team_reward is not None:
            info["team_vec_reward"] = team_reward
        else:
            info["team_vec_reward"] = vec_reward
        return info
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trading_env.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add envs/trading/orderbook_env.py tests/test_trading_env.py
git commit -m "feat: add multi-agent order book trading environment"
```

---

## Task 7: Patient Generation and Hospital State

**Files:**
- Create: `tests/test_hospital_components.py`
- Create: `envs/healthcare/patients.py`
- Create: `envs/healthcare/hospital.py`

**Step 1: Write failing tests**

```python
# tests/test_hospital_components.py
import numpy as np
import pytest
from envs.healthcare.patients import Patient, Acuity, PatientGenerator
from envs.healthcare.hospital import Hospital


class TestPatientGenerator:
    def test_generates_patients(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(
            arrival_rate=5.0,
            acuity_distribution=[0.6, 0.3, 0.1],
            rng=rng,
        )
        patients = gen.generate(timestep=0)
        assert isinstance(patients, list)
        assert all(isinstance(p, Patient) for p in patients)

    def test_acuity_distribution_over_many(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(arrival_rate=100.0, acuity_distribution=[0.6, 0.3, 0.1], rng=rng)
        patients = []
        for t in range(100):
            patients.extend(gen.generate(timestep=t))
        acuities = [p.acuity for p in patients]
        low_frac = sum(1 for a in acuities if a == Acuity.LOW) / len(acuities)
        assert 0.5 < low_frac < 0.7  # roughly 60%

    def test_patient_has_time_sensitivity(self):
        rng = np.random.default_rng(42)
        gen = PatientGenerator(arrival_rate=10.0, acuity_distribution=[0.6, 0.3, 0.1], rng=rng)
        patients = gen.generate(timestep=0)
        assert len(patients) > 0
        for p in patients:
            assert p.max_wait_time > 0
            assert p.length_of_stay > 0


class TestHospital:
    def test_admit_patient(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        assert h.admit(p, bed_type="general") is True
        assert h.occupancy("general") == 1

    def test_admit_full(self):
        h = Hospital(beds={"general": 1, "icu": 0, "emergency": 0})
        p1 = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        p2 = Patient(acuity=Acuity.ACUTE, arrival_time=0, max_wait_time=5, length_of_stay=10)
        assert h.admit(p1, bed_type="general") is True
        assert h.admit(p2, bed_type="general") is False

    def test_discharge_after_los(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.LOW, arrival_time=0, max_wait_time=10, length_of_stay=3)
        h.admit(p, bed_type="general")
        discharged = h.tick(current_time=1)
        assert len(discharged) == 0
        discharged = h.tick(current_time=3)
        assert len(discharged) == 1

    def test_capacity(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        assert h.capacity("general") == 5
        assert h.available("general") == 5

    def test_total_occupancy(self):
        h = Hospital(beds={"general": 5, "icu": 2, "emergency": 1})
        p = Patient(acuity=Acuity.CRITICAL, arrival_time=0, max_wait_time=2, length_of_stay=5)
        h.admit(p, bed_type="icu")
        assert h.total_occupancy() == 1
        assert h.total_capacity() == 8
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hospital_components.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement patients.py**

```python
# envs/healthcare/patients.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools

import numpy as np


class Acuity(Enum):
    LOW = 0       # can wait
    ACUTE = 1     # needs general bed
    CRITICAL = 2  # needs ICU


# Mapping from acuity to typical parameters
_ACUITY_PARAMS = {
    Acuity.LOW:      {"max_wait": (8, 24), "los": (4, 24)},
    Acuity.ACUTE:    {"max_wait": (2, 8),  "los": (12, 72)},
    Acuity.CRITICAL: {"max_wait": (1, 3),  "los": (24, 168)},
}

_patient_counter = itertools.count()


@dataclass
class Patient:
    acuity: Acuity
    arrival_time: int
    max_wait_time: int    # hours until outcome degrades
    length_of_stay: int   # hours of treatment needed
    patient_id: int = field(default_factory=lambda: next(_patient_counter))
    admitted_time: int | None = None

    @property
    def preferred_bed_type(self) -> str:
        if self.acuity == Acuity.CRITICAL:
            return "icu"
        if self.acuity == Acuity.ACUTE:
            return "general"
        return "general"


class PatientGenerator:
    """Generates patients with Poisson arrivals and configurable acuity distribution."""

    def __init__(
        self,
        arrival_rate: float,
        acuity_distribution: list[float],
        rng: np.random.Generator,
    ):
        self.arrival_rate = arrival_rate
        self.acuity_probs = np.array(acuity_distribution, dtype=np.float64)
        self.acuity_probs /= self.acuity_probs.sum()  # normalize
        self._rng = rng

    def generate(self, timestep: int) -> list[Patient]:
        n = self._rng.poisson(self.arrival_rate)
        patients = []
        for _ in range(n):
            acuity_idx = self._rng.choice(len(Acuity), p=self.acuity_probs)
            acuity = list(Acuity)[acuity_idx]
            params = _ACUITY_PARAMS[acuity]
            max_wait = int(self._rng.integers(params["max_wait"][0], params["max_wait"][1] + 1))
            los = int(self._rng.integers(params["los"][0], params["los"][1] + 1))
            patients.append(
                Patient(
                    acuity=acuity,
                    arrival_time=timestep,
                    max_wait_time=max_wait,
                    length_of_stay=los,
                )
            )
        return patients
```

**Step 4: Implement hospital.py**

```python
# envs/healthcare/hospital.py
from __future__ import annotations

from dataclasses import dataclass, field

from envs.healthcare.patients import Patient


class Hospital:
    """Manages bed inventory and patient treatment progression."""

    def __init__(self, beds: dict[str, int]):
        self._capacity = dict(beds)
        self._patients: dict[str, list[Patient]] = {bt: [] for bt in beds}

    def capacity(self, bed_type: str) -> int:
        return self._capacity[bed_type]

    def occupancy(self, bed_type: str) -> int:
        return len(self._patients[bed_type])

    def available(self, bed_type: str) -> int:
        return self._capacity[bed_type] - self.occupancy(bed_type)

    def total_capacity(self) -> int:
        return sum(self._capacity.values())

    def total_occupancy(self) -> int:
        return sum(len(ps) for ps in self._patients.values())

    def admit(self, patient: Patient, bed_type: str) -> bool:
        """Admit a patient to a bed. Returns False if no beds available."""
        if self.available(bed_type) <= 0:
            return False
        patient.admitted_time = patient.arrival_time  # set admission time
        self._patients[bed_type].append(patient)
        return True

    def tick(self, current_time: int) -> list[Patient]:
        """Advance time by one step. Discharge patients whose LOS is complete.
        Returns list of discharged patients."""
        discharged = []
        for bed_type in self._patients:
            remaining = []
            for p in self._patients[bed_type]:
                if p.admitted_time is not None and (current_time - p.admitted_time) >= p.length_of_stay:
                    discharged.append(p)
                else:
                    remaining.append(p)
            self._patients[bed_type] = remaining
        return discharged

    def get_patients(self, bed_type: str) -> list[Patient]:
        return list(self._patients[bed_type])

    def all_patients(self) -> list[Patient]:
        result = []
        for ps in self._patients.values():
            result.extend(ps)
        return result
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_hospital_components.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add envs/healthcare/patients.py envs/healthcare/hospital.py tests/test_hospital_components.py
git commit -m "feat: add patient generation and hospital state management"
```

---

## Task 8: Healthcare PettingZoo Environment

**Files:**
- Create: `tests/test_healthcare_env.py`
- Modify: `envs/healthcare/hospital_env.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_healthcare_env.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement healthcare environment**

```python
# envs/healthcare/hospital_env.py
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from envs.healthcare.hospital import Hospital
from envs.healthcare.patients import Acuity, Patient, PatientGenerator


# Actions per hospital:
# 0: accept next patient to preferred bed type
# 1: accept next patient to general bed
# 2: accept next patient to ICU
# 3: accept next patient to emergency
# 4: divert next patient to neighbour (round-robin)
# 5: discharge earliest stable patient
# 6: hold (do nothing)
NUM_ACTIONS = 7

# Max patients in queue for observation space
MAX_QUEUE_SIZE = 20


class HealthcareEnv(ParallelEnv):
    """Multi-agent hospital bed allocation environment.

    Each agent is a hospital deciding how to allocate limited beds.
    Rewards are 3-objective vectors: [mortality_reduction, equity, cost_efficiency].
    """

    metadata = {"name": "healthcare_v0"}

    def __init__(
        self,
        num_hospitals: int = 3,
        beds_per_hospital: dict[str, int] | None = None,
        num_objectives: int = 3,
        patient_arrival_rate: float = 2.0,
        acuity_distribution: list[float] | None = None,
        episode_length: int = 168,
    ):
        if beds_per_hospital is None:
            beds_per_hospital = {"general": 50, "icu": 10, "emergency": 5}
        if acuity_distribution is None:
            acuity_distribution = [0.6, 0.3, 0.1]

        self.num_hospitals = num_hospitals
        self.beds_config = beds_per_hospital
        self.num_objectives = num_objectives
        self.patient_arrival_rate = patient_arrival_rate
        self.acuity_distribution = acuity_distribution
        self.episode_length = episode_length

        self.possible_agents = [f"hospital_{i}" for i in range(num_hospitals)]
        self.agents = []

        self._hospitals: dict[str, Hospital] = {}
        self._queues: dict[str, list[Patient]] = {}
        self._generators: dict[str, PatientGenerator] = {}
        self._step_count = 0
        self._rng: np.random.Generator | None = None

        # Track metrics per step for reward computation
        self._patients_treated: dict[str, int] = {}
        self._patients_waiting_too_long: dict[str, int] = {}
        self._total_wait_times: dict[str, list[float]] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        # Per hospital: [general_occ, general_cap, icu_occ, icu_cap, emerg_occ, emerg_cap,
        #                queue_size, queue_acuity_counts(3), step/episode_length,
        #                neighbour_occupancies(num_hospitals-1)]
        size = 6 + 1 + 3 + 1 + (self.num_hospitals - 1)
        return Box(low=0, high=np.inf, shape=(size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        return Discrete(NUM_ACTIONS)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.agents = self.possible_agents[:]

        for a in self.agents:
            self._hospitals[a] = Hospital(beds=dict(self.beds_config))
            self._queues[a] = []
            self._generators[a] = PatientGenerator(
                arrival_rate=self.patient_arrival_rate,
                acuity_distribution=self.acuity_distribution,
                rng=np.random.default_rng(self._rng.integers(0, 2**32)),
            )
            self._patients_treated[a] = 0
            self._patients_waiting_too_long[a] = 0
            self._total_wait_times[a] = []

        # Generate initial patients
        for a in self.agents:
            self._queues[a] = self._generators[a].generate(timestep=0)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._get_info(a, np.zeros(self.num_objectives)) for a in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict, dict, dict, dict]:
        self._step_count += 1

        # Reset per-step counters
        step_treated = {a: 0 for a in self.agents}
        step_expired = {a: 0 for a in self.agents}
        step_diverted = {a: 0 for a in self.agents}

        # Process actions
        for agent, action in actions.items():
            hospital = self._hospitals[agent]
            queue = self._queues[agent]

            if action == 6 or not queue:  # hold or empty queue
                pass
            elif action == 0:  # accept to preferred bed type
                patient = queue[0]
                if hospital.admit(patient, patient.preferred_bed_type):
                    queue.pop(0)
                    step_treated[agent] += 1
            elif action in (1, 2, 3):  # accept to specific bed type
                bed_types = {1: "general", 2: "icu", 3: "emergency"}
                patient = queue[0]
                if hospital.admit(patient, bed_types[action]):
                    queue.pop(0)
                    step_treated[agent] += 1
            elif action == 4:  # divert to neighbour
                patient = queue.pop(0)
                neighbours = [a for a in self.agents if a != agent]
                if neighbours:
                    target = neighbours[self._step_count % len(neighbours)]
                    self._queues[target].append(patient)
                    step_diverted[agent] += 1
            elif action == 5:  # discharge earliest stable patient
                all_patients = hospital.all_patients()
                stable = [p for p in all_patients if p.acuity == Acuity.LOW]
                if stable:
                    # Force discharge by reducing LOS (simplified)
                    stable[0].length_of_stay = 0

        # Tick hospitals (discharge completed patients)
        for a in self.agents:
            self._hospitals[a].tick(current_time=self._step_count)

        # Check for patients who waited too long
        for a in self.agents:
            expired = []
            remaining = []
            for p in self._queues[a]:
                wait = self._step_count - p.arrival_time
                if wait >= p.max_wait_time:
                    expired.append(p)
                else:
                    remaining.append(p)
            step_expired[a] = len(expired)
            self._queues[a] = remaining
            self._patients_waiting_too_long[a] += len(expired)
            self._patients_treated[a] += step_treated[a]

        # Generate new patients
        for a in self.agents:
            new_patients = self._generators[a].generate(timestep=self._step_count)
            self._queues[a].extend(new_patients)

        # Compute rewards
        vec_rewards = {}
        for a in self.agents:
            vec_rewards[a] = self._compute_reward(a, step_treated[a], step_expired[a])

        team_reward = np.sum(list(vec_rewards.values()), axis=0)

        done = self._step_count >= self.episode_length
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: self._get_info(a, vec_rewards[a], team_reward) for a in self.agents}

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _compute_reward(
        self, agent: str, treated: int, expired: int
    ) -> np.ndarray:
        hospital = self._hospitals[agent]

        # Objective 1: Mortality reduction (treated vs expired this step)
        total_processed = treated + expired
        mortality_reduction = treated / max(total_processed, 1)

        # Objective 2: Equity (evenness of queue sizes across hospitals)
        queue_sizes = [len(self._queues[a]) for a in self.possible_agents if a in self._hospitals]
        if len(queue_sizes) > 1 and max(queue_sizes) > 0:
            mean_q = np.mean(queue_sizes)
            std_q = np.std(queue_sizes)
            equity = 1.0 - (std_q / max(mean_q, 1.0))  # 1 = perfectly equal
        else:
            equity = 1.0

        # Objective 3: Cost efficiency (utilization — not too empty, not too full)
        total_cap = hospital.total_capacity()
        total_occ = hospital.total_occupancy()
        utilization = total_occ / max(total_cap, 1)
        # Optimal utilization ~0.85: penalize both under- and over-utilization
        cost_efficiency = 1.0 - abs(utilization - 0.85) / 0.85

        return np.array([mortality_reduction, equity, cost_efficiency], dtype=np.float64)

    def _get_obs(self, agent: str) -> np.ndarray:
        hospital = self._hospitals[agent]
        queue = self._queues[agent]

        obs_parts = []
        for bt in ["general", "icu", "emergency"]:
            obs_parts.extend([hospital.occupancy(bt), hospital.capacity(bt)])

        obs_parts.append(len(queue))

        # Acuity counts in queue
        acuity_counts = [0, 0, 0]
        for p in queue[:MAX_QUEUE_SIZE]:
            acuity_counts[p.acuity.value] += 1
        obs_parts.extend(acuity_counts)

        # Normalized time
        obs_parts.append(self._step_count / max(self.episode_length, 1))

        # Neighbour occupancies (partial observability — just total occupancy ratios)
        for other in self.possible_agents:
            if other != agent and other in self._hospitals:
                h = self._hospitals[other]
                obs_parts.append(h.total_occupancy() / max(h.total_capacity(), 1))

        return np.array(obs_parts, dtype=np.float32)

    def _get_info(
        self, agent: str, vec_reward: np.ndarray, team_reward: np.ndarray | None = None
    ) -> dict:
        info = {"vec_reward": vec_reward}
        if team_reward is not None:
            info["team_vec_reward"] = team_reward
        else:
            info["team_vec_reward"] = vec_reward
        return info
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_healthcare_env.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add envs/healthcare/hospital_env.py tests/test_healthcare_env.py
git commit -m "feat: add multi-agent hospital bed allocation environment"
```

---

## Task 9: Multi-Agent MORL Agent Wrapper

**Files:**
- Create: `tests/test_agents.py`
- Create: `agents/mo_q_learning.py`

**Step 1: Write failing tests**

```python
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
        # With epsilon=0, should pick action with highest scalarized Q
        agent.q_table[0, 1, :] = [10.0, 10.0]
        action = agent.select_action(state=0, weights=np.array([0.5, 0.5]))
        assert action == 1

    def test_update(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2,
            learning_rate=1.0, gamma=0.0, epsilon=0.0, seed=42,
        )
        # With lr=1 and gamma=0, Q should become the reward exactly
        agent.update(
            state=0, action=1, reward=np.array([5.0, 3.0]),
            next_state=1, weights=np.array([0.5, 0.5]),
        )
        np.testing.assert_array_almost_equal(agent.q_table[0, 1], [5.0, 3.0])

    def test_epsilon_exploration(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2, epsilon=1.0, seed=42,
        )
        # With epsilon=1.0, all actions should be random
        actions = [agent.select_action(0, np.array([0.5, 0.5])) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # should explore multiple actions
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agents.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement tabular MO Q-Learning**

```python
# agents/mo_q_learning.py
from __future__ import annotations

import numpy as np


class TabularMOQLearning:
    """Multi-Objective Q-Learning with vector-valued Q-table.

    Uses scalarized action selection (weight vector applied at decision time)
    but maintains the full vector Q-values for multi-objective analysis.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_objectives: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

        # Q-table: shape (states, actions, objectives)
        self.q_table = np.zeros((num_states, num_actions, num_objectives), dtype=np.float64)

    def select_action(self, state: int, weights: np.ndarray) -> int:
        """Epsilon-greedy action selection using scalarized Q-values."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))

        # Scalarize Q-values with weights
        scalarized = self.q_table[state] @ weights  # shape (num_actions,)
        return int(np.argmax(scalarized))

    def update(
        self,
        state: int,
        action: int,
        reward: np.ndarray,
        next_state: int,
        weights: np.ndarray,
    ) -> None:
        """Update Q-values using the vector Bellman equation.

        Q(s,a) <- Q(s,a) + lr * (r + gamma * Q(s', a*) - Q(s,a))
        where a* = argmax_a' w . Q(s', a')
        """
        # Find best next action via scalarized Q
        scalarized_next = self.q_table[next_state] @ weights
        best_next_action = int(np.argmax(scalarized_next))

        # Vector TD update
        target = reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def get_pareto_q_values(self, state: int) -> np.ndarray:
        """Return Q-values for all actions at a state. Shape: (num_actions, num_objectives)."""
        return self.q_table[state].copy()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agents.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add agents/mo_q_learning.py tests/test_agents.py
git commit -m "feat: add tabular multi-objective Q-learning agent"
```

---

## Task 10: Experiment Runner

**Files:**
- Create: `tests/test_runner.py`
- Modify: `experiments/runner.py`
- Create: `experiments/configs/smoke_test.yaml`

**Step 1: Write failing tests**

```python
# tests/test_runner.py
import tempfile
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
        assert results["episode_returns"].shape[1] == 3  # 3 objectives

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: All tests FAIL with `ImportError`

**Step 3: Implement experiment runner**

```python
# experiments/runner.py
from __future__ import annotations

from dataclasses import dataclass, field
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
    """Load experiment configuration from YAML."""
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
    """Create the domain environment based on config."""
    if cfg.env == "trading":
        from envs.trading.orderbook_env import TradingEnv
        return TradingEnv(**cfg.env_params)
    elif cfg.env == "healthcare":
        from envs.healthcare.hospital_env import HealthcareEnv
        return HealthcareEnv(**cfg.env_params)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")


def _make_momas_config(cfg: ExperimentConfig, agents: list[str]) -> MOMASConfig:
    """Create MOMASConfig from experiment config."""
    # Default: equal weights across objectives
    weights = np.ones(cfg.num_objectives) / cfg.num_objectives

    if cfg.momas_utility_type == "team":
        utility_fns = {"shared": linear_utility(weights)}
    else:
        # Individual: each agent gets slightly different weights
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
    """Simple discretization for tabular agents: hash binned observations."""
    binned = np.clip(np.digitize(obs, np.linspace(obs.min() - 1, obs.max() + 1, num_bins)), 0, num_bins - 1)
    # Simple hash to get a state index
    h = 0
    for b in binned[:5]:  # use first 5 features to keep state space manageable
        h = h * num_bins + int(b)
    return h % 10000  # cap at 10000 states


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Run a single experiment and return results."""
    rng = np.random.default_rng(cfg.seed)

    # Create environment
    base_env = _make_env(cfg)
    base_env.reset(seed=cfg.seed)
    momas_cfg = _make_momas_config(cfg, base_env.possible_agents)
    env = MOMASWrapper(base_env, momas_cfg)

    # Create agents
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

    # Training loop
    all_episode_returns = []  # per episode, mean across agents

    for ep in range(cfg.num_episodes):
        obs, infos = env.reset(seed=int(rng.integers(0, 2**31)))
        episode_rewards = {a: np.zeros(cfg.num_objectives) for a in env.possible_agents}
        done = False

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

        # Mean return across agents for this episode
        mean_return = np.mean(list(episode_rewards.values()), axis=0)
        all_episode_returns.append(mean_return)

    episode_returns = np.array(all_episode_returns)

    # Evaluate
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

    # Save results
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
```

**Step 4: Create smoke test config**

```yaml
# experiments/configs/smoke_test.yaml
env: trading
env_params:
  num_agents: 2
  agent_types: [market_maker, momentum]
  episode_length: 20
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: SER
  num_objectives: 3
agent_type: tabular_moq
num_episodes: 3
seed: 42
output_dir: results
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add experiments/runner.py experiments/configs/smoke_test.yaml tests/test_runner.py
git commit -m "feat: add config-driven experiment runner with training loop and evaluation"
```

---

## Task 11: Full Sweep Experiment Configs

**Files:**
- Create: `experiments/configs/trading_team_team.yaml`
- Create: `experiments/configs/trading_team_social.yaml`
- Create: `experiments/configs/trading_team_individual.yaml`
- Create: `experiments/configs/trading_individual_social.yaml`
- Create: `experiments/configs/trading_individual_individual.yaml`
- Create: `experiments/configs/healthcare_team_team.yaml`
- Create: `experiments/configs/healthcare_team_social.yaml`
- Create: `experiments/configs/healthcare_team_individual.yaml`
- Create: `experiments/configs/healthcare_individual_social.yaml`
- Create: `experiments/configs/healthcare_individual_individual.yaml`
- Create: `experiments/sweep.py`

**Step 1: Create all 10 YAML config files**

Each file follows the same template, varying `env`, `env_params`, `reward_structure`, and `utility_type`. Trading configs use 4 agents with mixed types and 200 episode length. Healthcare configs use 3 hospitals and 168 episode length. All use 50 episodes and seed 42.

Example — `experiments/configs/trading_team_team.yaml`:
```yaml
env: trading
env_params:
  num_agents: 4
  agent_types: [market_maker, market_maker, momentum, arbitrageur]
  episode_length: 200
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: SER
  num_objectives: 3
agent_type: tabular_moq
num_episodes: 50
seed: 42
```

Repeat for all 10 combinations with appropriate `reward_structure` and `utility_type`.

**Step 2: Create sweep runner**

```python
# experiments/sweep.py
"""Run the full taxonomy sweep: 5 settings x 2 domains."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runner import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run MOMAS taxonomy sweep")
    parser.add_argument(
        "--configs-dir",
        default="experiments/configs",
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run configs matching this substring (e.g., 'trading' or 'team_team')",
    )
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    config_files = sorted(configs_dir.glob("*.yaml"))

    if args.filter:
        config_files = [f for f in config_files if args.filter in f.stem]

    print(f"Found {len(config_files)} configs to run")

    for config_file in config_files:
        if config_file.stem == "smoke_test":
            continue
        print(f"\n{'='*60}")
        print(f"Running: {config_file.stem}")
        print(f"{'='*60}")

        cfg = load_config(str(config_file))
        results = run_experiment(cfg)
        print(f"  Hypervolume: {results['hypervolume']:.4f}")
        print(f"  SER: {results['ser']:.4f}")
        print(f"  ESR: {results['esr']:.4f}")
        print(f"  Pareto front size: {len(results['pareto_front'])}")

    print("\nSweep complete!")


if __name__ == "__main__":
    main()
```

**Step 3: Test the sweep runner (smoke test)**

Run: `python -m experiments.sweep --filter smoke_test`
Expected: Runs the smoke_test config and prints results.

**Step 4: Commit**

```bash
git add experiments/configs/*.yaml experiments/sweep.py
git commit -m "feat: add full taxonomy sweep configs and runner (5 settings x 2 domains)"
```

---

## Task 12: Analysis Notebook

**Files:**
- Create: `notebooks/analysis.ipynb`

**Step 1: Create the analysis notebook**

Create a Jupyter notebook with cells for:

1. **Load results** — glob `results/**/*.npz` files, parse config from directory structure
2. **Pareto front plots** — 3D scatter plots per domain, coloured by taxonomy setting
3. **Hypervolume comparison** — bar chart comparing hypervolume across 5 settings for each domain
4. **SER vs ESR** — side-by-side bar chart showing how the two criteria differ
5. **Cross-domain comparison** — paired plots showing same taxonomy setting in trading vs healthcare

This is a visualization notebook — no complex logic, just loading `.npz` files and plotting with matplotlib/seaborn.

**Step 2: Commit**

```bash
git add notebooks/analysis.ipynb
git commit -m "feat: add analysis notebook for experiment visualization"
```

---

## Task 13: Integration Test — End-to-End Smoke Run

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
import tempfile
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
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All 10 parametrized tests PASS (2 domains x 5 taxonomy settings)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for all 10 taxonomy configurations"
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|----------------|
| 1 | Scaffolding | pyproject.toml, directory structure, dependencies |
| 2 | Utility functions | linear, threshold, chebyshev + welfare aggregators |
| 3 | Metrics | Pareto front, hypervolume, SER/ESR evaluation |
| 4 | MOMAS wrapper | MOMASConfig + MOMASWrapper for taxonomy settings |
| 5 | Order book | Limit order book matching engine |
| 6 | Trading env | PettingZoo multi-agent trading environment |
| 7 | Hospital components | Patient generation + hospital state |
| 8 | Healthcare env | PettingZoo multi-agent healthcare environment |
| 9 | MO Q-Learning agent | Tabular multi-objective agent |
| 10 | Experiment runner | Config-driven training + evaluation pipeline |
| 11 | Sweep configs | 10 YAML configs + sweep runner |
| 12 | Analysis notebook | Visualization of results |
| 13 | Integration tests | End-to-end smoke tests for all 10 configurations |
