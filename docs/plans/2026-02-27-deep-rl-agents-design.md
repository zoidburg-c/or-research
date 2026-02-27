# Deep RL Agents Design

**Date:** 2026-02-27
**Problem:** The only agent is a tabular MO Q-learner with a crude discretization hack. Continuous observation spaces (TradingEnv: 205 floats, HealthcareEnv: ~12 floats) need deep RL agents for meaningful learning.

## Agents

### 1. MO-DQN (`agents/mo_dqn.py`)

Custom PyTorch DQN for multi-objective rewards. Works directly on continuous observations.

- **Network:** 2-layer MLP (128, 128) → output `(num_actions * num_objectives)`, reshaped to `(num_actions, num_objectives)`
- **Action selection:** `select_action(obs, utility_fn=)` applies utility_fn to each action's Q-vector, epsilon-greedy
- **Update:** Standard DQN replay buffer (10k). Per-objective TD target on full Q-vector. Best next action via utility_fn scalarization. Target network soft-updated every 100 steps.
- **Constructor:** `MODQN(obs_dim, num_actions, num_objectives, lr=1e-3, gamma=0.99, epsilon=0.3, buffer_size=10000, batch_size=64, target_update_freq=100, seed=0)`

### 2. Envelope MOQ (`agents/envelope_moq.py`)

Thin wrapper around `morl_baselines.multi_policy.envelope.envelope.Envelope`.

- **Constructor:** Creates a mock gymnasium env with correct spaces + `reward_space` to satisfy Envelope init. Disables wandb logging.
- **Action selection:** `select_action(obs, utility_fn=, weights=)` — extracts weights or uses uniform, calls `agent.eval(obs, w)`
- **Update:** `update(obs, action, reward, next_obs, done, utility_fn=, weights=)` — pushes to replay buffer, calls `agent.update()` after learning_starts

## Environment Changes

Add `reward_space = Box(low=-inf, high=inf, shape=(num_objectives,))` to both `TradingEnv` and `HealthcareEnv` for mo_gymnasium compatibility.

## Runner Changes

- Agent dispatch for `"mo_dqn"` and `"envelope"` in `run_experiment()`
- Deep RL agents receive raw `obs` (np.ndarray), not discretized state ints
- Envelope update receives `done` flag (terminated or truncated)
- New YAML configs for deep RL agent sweeps

## Files Touched

| File | Change |
|------|--------|
| `agents/mo_dqn.py` | New — custom MO-DQN |
| `agents/envelope_moq.py` | New — Envelope wrapper |
| `envs/trading/orderbook_env.py` | Add `reward_space` |
| `envs/healthcare/hospital_env.py` | Add `reward_space` |
| `experiments/runner.py` | Agent dispatch, raw obs for deep RL |
| `experiments/configs/` | New configs |
| `tests/test_agents.py` | Unit tests for both agents |
| `tests/test_integration.py` | Agent type parametrization |
