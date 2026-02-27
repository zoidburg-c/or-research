# Deep RL Agents Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two deep RL agents (MO-DQN and Envelope MOQ wrapper) that work on continuous observations, replacing the crude discretization hack for meaningful learning.

**Architecture:** (1) Add `reward_space` to both envs for mo_gymnasium compatibility. (2) Implement a custom MO-DQN with PyTorch that outputs multi-objective Q-vectors and uses utility functions for action selection. (3) Wrap morl_baselines' Envelope agent with our API. (4) Update the runner to dispatch and handle deep RL agents with raw observations.

**Tech Stack:** Python, PyTorch, NumPy, morl_baselines, PettingZoo, pytest

---

### Task 1: Add reward_space to Environments

**Files:**
- Modify: `envs/healthcare/hospital_env.py`
- Modify: `envs/trading/orderbook_env.py`
- Test: `tests/test_healthcare_env.py`
- Test: `tests/test_trading_env.py`

**Step 1: Write failing tests**

Add to the end of `tests/test_healthcare_env.py`:

```python
    def test_has_reward_space(self):
        env = HealthcareEnv(num_hospitals=2)
        assert hasattr(env, "reward_space")
        assert env.reward_space.shape == (3,)
```

Add to the end of `tests/test_trading_env.py`:

```python
    def test_has_reward_space(self):
        env = TradingEnv(num_agents=2, agent_types=["market_maker", "momentum"])
        assert hasattr(env, "reward_space")
        assert env.reward_space.shape == (3,)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_healthcare_env.py::TestHealthcareEnvAPI::test_has_reward_space tests/test_trading_env.py::TestTradingEnvAPI::test_has_reward_space -v`
Expected: FAIL — `reward_space` not found

**Step 3: Add reward_space to both envs**

In `envs/healthcare/hospital_env.py`, add this line at the end of `__init__` (after `self._patients_waiting_too_long`):

```python
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float64)
```

In `envs/trading/orderbook_env.py`, add this line at the end of `__init__` (after all existing attribute assignments):

```python
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float64)
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add envs/healthcare/hospital_env.py envs/trading/orderbook_env.py tests/test_healthcare_env.py tests/test_trading_env.py
git commit -m "feat: add reward_space to environments for mo_gymnasium compatibility

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Implement MO-DQN Agent

**Files:**
- Create: `agents/mo_dqn.py`
- Test: `tests/test_mo_dqn.py`

**Step 1: Write failing tests**

Create `tests/test_mo_dqn.py`:

```python
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
        # Just verify both return valid actions (network is random-init so actions may differ)
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
        # Fill buffer with enough transitions to trigger a batch update
        for i in range(10):
            next_obs = np.ones(4, dtype=np.float32) * i
            agent.update(obs, i % 3, np.array([1.0, 2.0]), next_obs, done=False, utility_fn=u)
            obs = next_obs

    def test_update_changes_q_values(self):
        agent = MODQN(obs_dim=4, num_actions=3, num_objectives=2,
                      buffer_size=100, batch_size=4, lr=0.01, seed=42)
        u = linear_utility(np.array([0.5, 0.5]))
        obs = np.zeros(4, dtype=np.float32)

        # Get initial Q-values
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_before = agent.q_net(obs_t).reshape(agent.num_actions, agent.num_objectives).clone()

        # Train with consistent positive reward
        for i in range(20):
            agent.update(obs, 0, np.array([10.0, 10.0]), obs, done=False, utility_fn=u)

        with torch.no_grad():
            q_after = agent.q_net(obs_t).reshape(agent.num_actions, agent.num_objectives)

        # Q-values should change after training
        assert not torch.allclose(q_before, q_after)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_mo_dqn.py -v`
Expected: FAIL — module `agents.mo_dqn` not found

**Step 3: Implement MO-DQN**

Create `agents/mo_dqn.py`:

```python
from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _QNetwork(nn.Module):
    """MLP that outputs (num_actions * num_objectives) values."""

    def __init__(self, obs_dim: int, num_actions: int, num_objectives: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions * num_objectives),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MODQN:
    """Multi-Objective DQN for continuous observation spaces.

    Outputs vector Q-values per action and uses a utility function
    for scalarized action selection.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_objectives: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        self.q_net = _QNetwork(obs_dim, num_actions, num_objectives)
        self.target_net = _QNetwork(obs_dim, num_actions, num_objectives)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self._buffer: deque[tuple] = deque(maxlen=buffer_size)
        self._step_count = 0

    def select_action(self, obs: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_flat = self.q_net(obs_t)
            q_values = q_flat.reshape(self.num_actions, self.num_objectives).numpy()

        if utility_fn is not None:
            scalarized = np.array([utility_fn(q_values[a]) for a in range(self.num_actions)])
        elif weights is not None:
            scalarized = q_values @ weights
        else:
            scalarized = q_values.sum(axis=1)

        return int(np.argmax(scalarized))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: bool = False,
        utility_fn: Callable | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self._buffer.append((obs.copy(), action, reward.copy(), next_obs.copy(), done))
        self._step_count += 1

        if len(self._buffer) < self.batch_size:
            return

        # Sample batch
        indices = self._rng.integers(0, len(self._buffer), size=self.batch_size)
        batch = [self._buffer[i] for i in indices]
        obs_b = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        act_b = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rew_b = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32)
        next_obs_b = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        done_b = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        # Current Q-values for taken actions: (batch, num_objectives)
        q_flat = self.q_net(obs_b)
        q_all = q_flat.reshape(self.batch_size, self.num_actions, self.num_objectives)
        q_current = q_all[torch.arange(self.batch_size), act_b]

        # Target: best next action via utility scalarization
        with torch.no_grad():
            next_q_flat = self.target_net(next_obs_b)
            next_q_all = next_q_flat.reshape(self.batch_size, self.num_actions, self.num_objectives)
            next_q_np = next_q_all.numpy()

            best_next_actions = np.zeros(self.batch_size, dtype=np.int64)
            for i in range(self.batch_size):
                if utility_fn is not None:
                    scores = np.array([utility_fn(next_q_np[i, a]) for a in range(self.num_actions)])
                elif weights is not None:
                    scores = next_q_np[i] @ weights
                else:
                    scores = next_q_np[i].sum(axis=1)
                best_next_actions[i] = np.argmax(scores)

            best_next = next_q_all[torch.arange(self.batch_size), torch.tensor(best_next_actions)]
            target = rew_b + self.gamma * (1 - done_b.unsqueeze(1)) * best_next

        loss = nn.functional.mse_loss(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_mo_dqn.py tests/test_agents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/mo_dqn.py tests/test_mo_dqn.py
git commit -m "feat: add MO-DQN agent for continuous observation spaces

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Implement Envelope MOQ Wrapper

**Files:**
- Create: `agents/envelope_moq.py`
- Test: `tests/test_envelope_moq.py`

**Step 1: Write failing tests**

Create `tests/test_envelope_moq.py`:

```python
import numpy as np
import pytest
from agents.envelope_moq import EnvelopeMOQ
from momas.utility import linear_utility


class TestEnvelopeMOQ:
    def test_init(self):
        agent = EnvelopeMOQ(obs_dim=12, num_actions=7, num_objectives=3, seed=42)
        assert agent.obs_dim == 12
        assert agent.num_actions == 7
        assert agent.num_objectives == 3

    def test_select_action_shape(self):
        agent = EnvelopeMOQ(obs_dim=12, num_actions=7, num_objectives=3, seed=42)
        obs = np.random.default_rng(42).standard_normal(12).astype(np.float32)
        u = linear_utility(np.array([1/3, 1/3, 1/3]))
        action = agent.select_action(obs, utility_fn=u)
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_select_action_with_weights(self):
        agent = EnvelopeMOQ(obs_dim=4, num_actions=3, num_objectives=2, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        w = np.array([0.5, 0.5])
        action = agent.select_action(obs, weights=w)
        assert 0 <= action < 3

    def test_update_runs_without_error(self):
        agent = EnvelopeMOQ(obs_dim=4, num_actions=3, num_objectives=2,
                            buffer_size=100, batch_size=4, learning_starts=5, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        w = np.array([0.5, 0.5])
        for i in range(10):
            next_obs = np.ones(4, dtype=np.float32) * i * 0.1
            agent.update(obs, i % 3, np.array([1.0, 2.0]), next_obs, done=False, weights=w)
            obs = next_obs
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_envelope_moq.py -v`
Expected: FAIL — module not found

**Step 3: Implement Envelope wrapper**

Create `agents/envelope_moq.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class EnvelopeMOQ:
    """Wrapper around morl_baselines Envelope for multi-objective Q-learning.

    Creates a mock gymnasium env to satisfy Envelope's constructor,
    then exposes our standard agent API.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_objectives: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        learning_starts: int = 100,
        target_update_freq: int = 200,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_objectives = num_objectives
        self._default_weights = np.ones(num_objectives, dtype=np.float32) / num_objectives

        # Create mock env to satisfy Envelope constructor
        mock_env = _MockEnv(obs_dim, num_actions, num_objectives)

        from morl_baselines.multi_policy.envelope.envelope import Envelope

        self._agent = Envelope(
            env=mock_env,
            learning_rate=lr,
            initial_epsilon=epsilon,
            final_epsilon=epsilon,
            tau=1.0,
            target_net_update_freq=target_update_freq,
            buffer_size=buffer_size,
            net_arch=[128, 128],
            batch_size=batch_size,
            learning_starts=learning_starts,
            gamma=gamma,
            envelope=True,
            log=False,
            seed=seed,
        )

    def select_action(self, obs: np.ndarray, utility_fn: Callable | None = None, weights: np.ndarray | None = None) -> int:
        w = self._resolve_weights(utility_fn, weights)
        return int(self._agent.eval(np.asarray(obs, dtype=np.float32), w))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: bool = False,
        utility_fn: Callable | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        self._agent.replay_buffer.add(
            np.asarray(obs, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray([action]),
            np.asarray(reward, dtype=np.float32),
            np.asarray([done], dtype=np.float32),
        )
        self._agent.global_step += 1
        if self._agent.global_step >= self._agent.learning_starts and len(self._agent.replay_buffer) >= self._agent.batch_size:
            self._agent.update()

    def _resolve_weights(self, utility_fn: Callable | None, weights: np.ndarray | None) -> np.ndarray:
        if weights is not None:
            return np.asarray(weights, dtype=np.float32)
        return self._default_weights.copy()


class _MockEnv(gym.Env):
    """Minimal gymnasium env to satisfy morl_baselines constructor."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, num_actions: int, num_objectives: int):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Discrete(num_actions)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), np.zeros(self.reward_space.shape[0]), False, False, {}
```

**Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_envelope_moq.py tests/test_mo_dqn.py tests/test_agents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/envelope_moq.py tests/test_envelope_moq.py
git commit -m "feat: add Envelope MOQ wrapper around morl_baselines

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Update Runner for Deep RL Agent Dispatch

**Files:**
- Modify: `experiments/runner.py`
- Test: `tests/test_runner.py`

**Step 1: Write failing tests**

Add to the end of `tests/test_runner.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_runner.py::TestDeepRLAgents -v`
Expected: FAIL — runner doesn't know about mo_dqn or envelope agent types

**Step 3: Update runner to dispatch deep RL agents**

In `experiments/runner.py`, add imports at the top (after the existing imports):

```python
from agents.mo_dqn import MODQN
from agents.envelope_moq import EnvelopeMOQ
```

Replace the agent creation block and training loop in `run_experiment`. The key changes are:

1. Agent creation dispatches based on `cfg.agent_type`
2. Deep RL agents get raw obs (np.ndarray) not discretized ints
3. Deep RL agents need `done` flag in update
4. Action selection and update pass `utility_fn` for all agent types

Replace the `run_experiment` function entirely with:

```python
def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    setting_hash = hash(cfg.momas_utility_type) % (2**31)
    exp_seed = (cfg.seed + setting_hash) % (2**31)
    rng = np.random.default_rng(exp_seed)

    base_env = _make_env(cfg)
    base_env.reset(seed=cfg.seed)
    momas_cfg, agent_weights = _make_momas_config(cfg, base_env.possible_agents)
    env = MOMASWrapper(base_env, momas_cfg)

    is_deep_rl = cfg.agent_type in ("mo_dqn", "envelope")

    agents = {}
    for a in env.possible_agents:
        obs_dim = env.observation_space(a).shape[0]
        n_actions = env.action_space(a).n
        agent_seed = int(rng.integers(0, 2**31))

        if cfg.agent_type == "mo_dqn":
            agents[a] = MODQN(
                obs_dim=obs_dim,
                num_actions=n_actions,
                num_objectives=cfg.num_objectives,
                lr=1e-3,
                gamma=0.99,
                epsilon=0.3,
                buffer_size=10_000,
                batch_size=64,
                target_update_freq=100,
                seed=agent_seed,
            )
        elif cfg.agent_type == "envelope":
            agents[a] = EnvelopeMOQ(
                obs_dim=obs_dim,
                num_actions=n_actions,
                num_objectives=cfg.num_objectives,
                lr=3e-4,
                gamma=0.99,
                epsilon=0.3,
                buffer_size=10_000,
                batch_size=64,
                learning_starts=100,
                target_update_freq=200,
                seed=agent_seed,
            )
        else:
            agents[a] = TabularMOQLearning(
                num_states=10000,
                num_actions=n_actions,
                num_objectives=cfg.num_objectives,
                learning_rate=0.1,
                gamma=0.99,
                epsilon=0.3,
                seed=agent_seed,
            )

    all_episode_returns = []

    for ep in range(cfg.num_episodes):
        obs, infos = env.reset(seed=int(rng.integers(0, 2**31)))
        episode_rewards = {a: np.zeros(cfg.num_objectives) for a in env.possible_agents}

        while env.agents:
            actions = {}
            prev_obs = {}
            states = {}
            for a in env.agents:
                prev_obs[a] = obs[a]
                if not is_deep_rl:
                    s = _discretize_obs(obs[a])
                    states[a] = s
                u_fn = _get_agent_utility(momas_cfg, a)
                if is_deep_rl:
                    actions[a] = agents[a].select_action(obs[a], utility_fn=u_fn)
                else:
                    actions[a] = agents[a].select_action(states[a], utility_fn=u_fn)

            obs, rewards, terms, truncs, infos = env.step(actions)

            for a in prev_obs:
                if a not in infos:
                    continue
                vec_r = infos[a]["vec_reward"]
                episode_rewards[a] += vec_r

                shaped_r = _weight_reward(vec_r, agent_weights[a]) if cfg.momas_utility_type != "team" else vec_r

                if cfg.momas_utility_type == "social_choice" and momas_cfg.welfare_function is not None:
                    agent_utilities = []
                    for other_a in prev_obs:
                        if other_a in infos:
                            other_u_fn = _get_agent_utility(momas_cfg, other_a)
                            other_vec_r = infos[other_a]["vec_reward"]
                            agent_utilities.append(other_u_fn(other_vec_r))
                    welfare = momas_cfg.welfare_function(agent_utilities)
                    own_utility = _get_agent_utility(momas_cfg, a)(vec_r)
                    if abs(own_utility) > 1e-10:
                        welfare_scale = welfare / (len(prev_obs) * own_utility)
                        shaped_r = shaped_r * welfare_scale

                u_fn = _get_agent_utility(momas_cfg, a)
                done = terms.get(a, False) or truncs.get(a, False)

                if is_deep_rl:
                    next_o = obs[a] if a in obs else prev_obs[a]
                    agents[a].update(prev_obs[a], actions[a], shaped_r, next_o,
                                     done=done, utility_fn=u_fn)
                else:
                    next_s = _discretize_obs(obs[a]) if a in obs else states[a]
                    agents[a].update(states[a], actions[a], shaped_r, next_s, utility_fn=u_fn)

        mean_return = np.mean(list(episode_rewards.values()), axis=0)
        all_episode_returns.append(mean_return)

    episode_returns = np.array(all_episode_returns)

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
git commit -m "feat: runner dispatches MO-DQN and Envelope agents with raw observations

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add Deep RL Experiment Configs

**Files:**
- Create: `experiments/configs/healthcare_team_team_dqn.yaml`
- Create: `experiments/configs/trading_team_team_dqn.yaml`
- Create: `experiments/configs/healthcare_team_team_envelope.yaml`
- Create: `experiments/configs/trading_team_team_envelope.yaml`

**Step 1: Create configs**

Create `experiments/configs/healthcare_team_team_dqn.yaml`:

```yaml
env: healthcare
env_params:
  num_hospitals: 3
  episode_length: 168
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: SER
  num_objectives: 3
agent_type: mo_dqn
num_episodes: 50
seed: 42
```

Create `experiments/configs/trading_team_team_dqn.yaml`:

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
agent_type: mo_dqn
num_episodes: 50
seed: 42
```

Create `experiments/configs/healthcare_team_team_envelope.yaml`:

```yaml
env: healthcare
env_params:
  num_hospitals: 3
  episode_length: 168
momas:
  reward_structure: team
  utility_type: team
  optimisation_criterion: SER
  num_objectives: 3
agent_type: envelope
num_episodes: 50
seed: 42
```

Create `experiments/configs/trading_team_team_envelope.yaml`:

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
agent_type: envelope
num_episodes: 50
seed: 42
```

**Step 2: Commit**

```bash
git add experiments/configs/healthcare_team_team_dqn.yaml experiments/configs/trading_team_team_dqn.yaml experiments/configs/healthcare_team_team_envelope.yaml experiments/configs/trading_team_team_envelope.yaml
git commit -m "feat: add experiment configs for MO-DQN and Envelope agents

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Integration Tests for Deep RL Agents

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Add agent_type parametrization**

Add a new parametrize decorator and update the test method. The full updated test class should be:

```python
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

    @pytest.mark.parametrize("env_name,env_params", [
        ("healthcare", {"num_hospitals": 2, "episode_length": 5}),
    ])
    @pytest.mark.parametrize("agent_type", ["mo_dqn", "envelope"])
    def test_deep_rl_agents(self, env_name, env_params, agent_type, output_dir):
        cfg = ExperimentConfig(
            env=env_name,
            env_params=env_params,
            momas_reward_structure="team",
            momas_utility_type="team",
            momas_criterion="SER",
            num_objectives=3,
            agent_type=agent_type,
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
Expected: ALL PASS (20 tabular + 2 deep RL = 22 tests)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add deep RL agent integration tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
