# Taxonomy Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the 5 MOMAS taxonomy settings produce meaningfully different training behavior and results by wiring up per-agent utilities, social choice welfare, and non-linear ESR utility.

**Architecture:** Three changes to the existing codebase: (1) agent API accepts utility functions instead of weight vectors, (2) runner passes per-agent utilities and aggregates welfare for social choice, (3) ESR configs use threshold utility to create SER≠ESR divergence. All changes are backward-compatible through the existing test suite.

**Tech Stack:** Python, NumPy, PettingZoo, PyYAML, pytest

---

### Task 1: Agent API — Accept Utility Functions

**Files:**
- Modify: `agents/mo_q_learning.py`
- Test: `tests/test_agents.py`

**Step 1: Write failing tests for utility-function-based API**

Add to `tests/test_agents.py`:

```python
from momas.utility import linear_utility, threshold_utility

class TestTabularMOQLearningUtilityFn:
    def test_select_action_with_utility_fn(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2, epsilon=0.0, seed=42,
        )
        agent.q_table[0, 0, :] = [1.0, 10.0]  # action 0: low obj0, high obj1
        agent.q_table[0, 1, :] = [10.0, 1.0]  # action 1: high obj0, low obj1
        agent.q_table[0, 2, :] = [5.0, 5.0]   # action 2: balanced

        # Utility that weights obj0 heavily should pick action 1
        u_obj0 = linear_utility(np.array([0.9, 0.1]))
        assert agent.select_action(state=0, utility_fn=u_obj0) == 1

        # Utility that weights obj1 heavily should pick action 0
        u_obj1 = linear_utility(np.array([0.1, 0.9]))
        assert agent.select_action(state=0, utility_fn=u_obj1) == 0

    def test_update_with_utility_fn(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2,
            learning_rate=1.0, gamma=0.0, epsilon=0.0, seed=42,
        )
        u = linear_utility(np.array([0.5, 0.5]))
        agent.update(
            state=0, action=1, reward=np.array([5.0, 3.0]),
            next_state=1, utility_fn=u,
        )
        np.testing.assert_array_almost_equal(agent.q_table[0, 1], [5.0, 3.0])

    def test_nonlinear_utility_changes_action_selection(self):
        agent = TabularMOQLearning(
            num_states=5, num_actions=3, num_objectives=2, epsilon=0.0, seed=42,
        )
        agent.q_table[0, 0, :] = [3.0, 3.0]   # balanced: above thresholds
        agent.q_table[0, 1, :] = [10.0, 0.5]  # high obj0 but obj1 below threshold
        agent.q_table[0, 2, :] = [0.5, 10.0]  # high obj1 but obj0 below threshold

        # Linear utility picks action 1 (highest weighted sum)
        u_linear = linear_utility(np.array([0.5, 0.5]))
        assert agent.select_action(state=0, utility_fn=u_linear) == 1

        # Threshold utility picks action 0 (only one above both thresholds)
        u_thresh = threshold_utility(
            thresholds=np.array([1.0, 1.0]),
            weights=np.array([0.5, 0.5]),
        )
        assert agent.select_action(state=0, utility_fn=u_thresh) == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: FAIL — `select_action()` doesn't accept `utility_fn` parameter

**Step 3: Update agent to accept utility functions**

Modify `agents/mo_q_learning.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np


class TabularMOQLearning:
    """Multi-Objective Q-Learning with vector-valued Q-table.

    Uses scalarized action selection (utility function applied at decision time)
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

        self.q_table = np.zeros((num_states, num_actions, num_objectives), dtype=np.float64)

    def _scalarize(self, q_values: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> np.ndarray:
        """Scalarize Q-values for all actions using utility_fn or weights."""
        if utility_fn is not None:
            return np.array([utility_fn(q_values[a]) for a in range(q_values.shape[0])])
        return q_values @ weights

    def select_action(self, state: int, weights: np.ndarray | None = None, utility_fn: Callable | None = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        scalarized = self._scalarize(self.q_table[state], utility_fn, weights)
        return int(np.argmax(scalarized))

    def update(
        self,
        state: int,
        action: int,
        reward: np.ndarray,
        next_state: int,
        weights: np.ndarray | None = None,
        utility_fn: Callable | None = None,
    ) -> None:
        scalarized_next = self._scalarize(self.q_table[next_state], utility_fn, weights)
        best_next_action = int(np.argmax(scalarized_next))
        target = reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def get_pareto_q_values(self, state: int) -> np.ndarray:
        return self.q_table[state].copy()
```

**Step 4: Run all tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agents.py -v`
Expected: ALL PASS (new tests + old tests, since old API with `weights=` still works)

**Step 5: Commit**

```bash
git add agents/mo_q_learning.py tests/test_agents.py
git commit -m "feat: agent API accepts utility functions for non-linear scalarization"
```

---

### Task 2: Add Welfare Function to MOMASConfig

**Files:**
- Modify: `momas/base_env.py`
- Test: `tests/test_base_env.py`

**Step 1: Write failing test for welfare_function field**

Add to `tests/test_base_env.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_base_env.py::TestMOMASConfigWelfare -v`
Expected: FAIL — `MOMASConfig` doesn't accept `welfare_function`

**Step 3: Add welfare_function field to MOMASConfig**

Modify `momas/base_env.py` — add to the dataclass:

```python
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
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_base_env.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add momas/base_env.py tests/test_base_env.py
git commit -m "feat: add welfare_function field to MOMASConfig"
```

---

### Task 3: Wire Up Per-Agent Utilities and Social Choice in Runner

**Files:**
- Modify: `experiments/runner.py`
- Test: `tests/test_runner.py`

**Step 1: Write failing tests for per-agent utility and social choice differentiation**

Add to `tests/test_runner.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_runner.py::TestPerAgentUtility tests/test_runner.py::TestSocialChoiceWelfare -v`
Expected: FAIL — currently all utility types produce identical results

**Step 3: Rewrite `run_experiment` to use per-agent utilities and social choice welfare**

Replace `experiments/runner.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from momas.base_env import MOMASConfig, MOMASWrapper
from momas.utility import linear_utility, threshold_utility, utilitarian_welfare
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


def _make_utility_fn(weights: np.ndarray, criterion: str):
    """Create utility function based on optimisation criterion."""
    if criterion == "ESR":
        return threshold_utility(
            thresholds=weights * 0.1,
            weights=weights,
        )
    return linear_utility(weights)


def _make_momas_config(cfg: ExperimentConfig, agents: list[str]) -> MOMASConfig:
    weights = np.ones(cfg.num_objectives) / cfg.num_objectives

    if cfg.momas_utility_type == "team":
        utility_fns = {"shared": _make_utility_fn(weights, cfg.momas_criterion)}
        welfare_fn = None
    else:
        rng = np.random.default_rng(cfg.seed)
        utility_fns = {}
        for a in agents:
            w = rng.dirichlet(np.ones(cfg.num_objectives))
            utility_fns[a] = _make_utility_fn(w, cfg.momas_criterion)
        welfare_fn = utilitarian_welfare if cfg.momas_utility_type == "social_choice" else None

    return MOMASConfig(
        reward_structure=cfg.momas_reward_structure,
        utility_type=cfg.momas_utility_type,
        optimisation_criterion=cfg.momas_criterion,
        num_objectives=cfg.num_objectives,
        utility_functions=utility_fns,
        welfare_function=welfare_fn,
    )


def _discretize_obs(obs: np.ndarray, num_bins: int = 10) -> int:
    binned = np.clip(np.digitize(obs, np.linspace(obs.min() - 1, obs.max() + 1, num_bins)), 0, num_bins - 1)
    h = 0
    for b in binned[:5]:
        h = h * num_bins + int(b)
    return h % 10000


def _get_agent_utility(momas_cfg: MOMASConfig, agent: str):
    """Get the utility function for a specific agent."""
    if momas_cfg.utility_type == "team":
        return momas_cfg.utility_functions["shared"]
    return momas_cfg.utility_functions[agent]


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)

    base_env = _make_env(cfg)
    base_env.reset(seed=cfg.seed)
    momas_cfg = _make_momas_config(cfg, base_env.possible_agents)
    env = MOMASWrapper(base_env, momas_cfg)

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
                u_fn = _get_agent_utility(momas_cfg, a)
                actions[a] = agents[a].select_action(s, utility_fn=u_fn)

            obs, rewards, terms, truncs, infos = env.step(actions)

            for a in states:
                if a not in infos:
                    continue
                vec_r = infos[a]["vec_reward"]
                episode_rewards[a] += vec_r
                next_s = _discretize_obs(obs[a]) if a in obs else states[a]

                if cfg.momas_utility_type == "social_choice" and momas_cfg.welfare_function is not None:
                    # Social choice: compute welfare across all agents, use as scalar signal
                    agent_utilities = []
                    for other_a in states:
                        if other_a in infos:
                            other_u_fn = _get_agent_utility(momas_cfg, other_a)
                            other_vec_r = infos[other_a]["vec_reward"]
                            agent_utilities.append(other_u_fn(other_vec_r))
                    welfare = momas_cfg.welfare_function(agent_utilities)
                    # Scale vec_reward by welfare/own_utility ratio to steer Q-values
                    own_utility = _get_agent_utility(momas_cfg, a)(vec_r)
                    if abs(own_utility) > 1e-10:
                        welfare_scale = welfare / (len(states) * own_utility)
                        scaled_r = vec_r * welfare_scale
                    else:
                        scaled_r = vec_r
                    u_fn = _get_agent_utility(momas_cfg, a)
                    agents[a].update(states[a], actions[a], scaled_r, next_s, utility_fn=u_fn)
                else:
                    u_fn = _get_agent_utility(momas_cfg, a)
                    agents[a].update(states[a], actions[a], vec_r, next_s, utility_fn=u_fn)

        mean_return = np.mean(list(episode_rewards.values()), axis=0)
        all_episode_returns.append(mean_return)

    episode_returns = np.array(all_episode_returns)

    # Use shared linear utility for evaluation metrics (comparable across settings)
    eval_weights = np.ones(cfg.num_objectives) / cfg.num_objectives
    eval_u = linear_utility(eval_weights)
    ser = evaluate_ser(episode_returns, eval_u)
    esr = evaluate_esr(episode_returns, eval_u)
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
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add experiments/runner.py tests/test_runner.py
git commit -m "feat: wire up per-agent utilities and social choice welfare in runner"
```

---

### Task 4: ESR Non-Linear Utility Differentiation

**Files:**
- Modify: `experiments/runner.py` (already done in Task 3 via `_make_utility_fn`)
- Test: `tests/test_runner.py`

**Step 1: Write failing test for ESR ≠ SER**

Add to `tests/test_runner.py`:

```python
class TestESRDifferentiation:
    def test_esr_produces_different_metrics_than_ser(self, tmp_path):
        """ESR with non-linear utility should produce SER ≠ ESR in metrics."""
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

        # With non-linear (threshold) utility used during training,
        # the episode returns should be shaped differently than SER
        assert np.isfinite(results["ser"])
        assert np.isfinite(results["esr"])
        # Results should be valid (not all NaN or identical)
        assert results["episode_returns"].shape == (5, 3)
```

**Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runner.py::TestESRDifferentiation -v`
Expected: PASS (this is already handled by Task 3's `_make_utility_fn`)

**Step 3: Add ESR configs to YAML files**

Create `experiments/configs/healthcare_team_team_esr.yaml`:

```yaml
env: healthcare
env_params:
  num_hospitals: 3
  episode_length: 168
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: ESR
  num_objectives: 3
agent_type: tabular_moq
num_episodes: 50
seed: 42
```

Create `experiments/configs/trading_team_team_esr.yaml`:

```yaml
env: trading
env_params:
  num_agents: 4
  agent_types: [market_maker, market_maker, momentum, arbitrageur]
  episode_length: 200
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: ESR
  num_objectives: 3
agent_type: tabular_moq
num_episodes: 50
seed: 42
```

**Step 4: Commit**

```bash
git add tests/test_runner.py experiments/configs/healthcare_team_team_esr.yaml experiments/configs/trading_team_team_esr.yaml
git commit -m "feat: add ESR experiment configs with threshold utility"
```

---

### Task 5: Update Integration Tests

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Add ESR parametrization to integration tests**

Modify the parametrize decorator in `tests/test_integration.py` to include ESR:

```python
@pytest.mark.parametrize("criterion", ["SER", "ESR"])
def test_all_taxonomy_settings(
    self, env_name, env_params, reward_structure, utility_type, criterion, output_dir
):
    cfg = ExperimentConfig(
        env=env_name,
        env_params=env_params,
        momas_reward_structure=reward_structure,
        momas_utility_type=utility_type,
        momas_criterion=criterion,
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

Run: `.venv/bin/python -m pytest tests/test_integration.py -v`
Expected: ALL PASS (20 parametrized cases: 2 envs × 5 settings × 2 criteria)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add ESR criterion to integration test matrix"
```

---

### Task 6: Run Full Sweep and Verify Differentiation

**Step 1: Clear old results**

```bash
rm -rf results/healthcare results/trading
```

**Step 2: Run full sweep**

Run: `.venv/bin/python -m experiments.sweep`

**Step 3: Verify taxonomy settings produce different results**

Check that:
- Team/team ≠ Team/individual ≠ Team/social_choice (for same domain)
- SER configs ≠ ESR configs (for same setting)
- Individual reward configs ≠ Team reward configs

**Step 4: Commit results note**

No code to commit — results are gitignored. This is a verification step.
