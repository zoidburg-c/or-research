# Fix Taxonomy Gaps: Per-Agent Utilities, Social Choice Welfare, ESR Training

**Date:** 2026-02-27
**Problem:** All 5 taxonomy settings produce identical results because the runner ignores utility type, social choice welfare, and ESR criterion during training.

## Changes

### 1. Per-Agent Utility-Aware Action Selection

**Files:** `agents/mo_q_learning.py`, `experiments/runner.py`

The agent API changes from weight-vector-based to utility-function-based scalarization:

- `select_action(state, utility_fn)` — applies `utility_fn` to each action's Q-vector, picks argmax
- `update(state, action, reward, next_state, utility_fn)` — uses `utility_fn` to select best next action for Q-target

The runner extracts per-agent utility functions from `MOMASConfig.utility_functions` and passes each agent its own function. For "team" utility, all agents share one function. For "individual"/"social_choice", each agent gets its own Dirichlet-weighted utility.

### 2. Social Choice Welfare in Training Loop

**Files:** `experiments/runner.py`, `momas/base_env.py`, social choice YAML configs

In social choice mode, Q-updates use a welfare-aggregated signal:

1. Each agent computes its scalar utility from its vec_reward
2. `utilitarian_welfare` (configurable) aggregates all agents' utilities into one scalar
3. This welfare scalar replaces per-agent scalarized reward in Q-updates
4. All agents learn to optimize collective welfare, not just their own utility

`MOMASConfig` gains an optional `welfare_function` field (default: `utilitarian_welfare`).

### 3. Non-Linear Utility for ESR Configs

**Files:** `experiments/runner.py`, ESR YAML configs

When `optimisation_criterion == "ESR"`, the runner constructs `threshold_utility` instead of `linear_utility`. This makes SER ≠ ESR since the two criteria diverge for non-linear utility functions.

Thresholds are set at conservative values (close to 0) so the utility isn't dominated by -inf returns but still creates meaningful non-linearity.

YAML configs gain a `utility_function_type` field: `linear` (default/SER) or `threshold` (ESR).

### 4. Tests

- Per-agent utility functions produce different action selections given different weights
- Social choice welfare aggregation changes Q-update signal vs individual utility
- ESR configs with threshold utility produce SER ≠ ESR in post-hoc metrics

### 5. Files Touched

| File | Change |
|------|--------|
| `agents/mo_q_learning.py` | `select_action`/`update` accept `utility_fn` callable |
| `experiments/runner.py` | Per-agent utilities, social choice welfare loop, ESR utility selection |
| `momas/base_env.py` | Add optional `welfare_function` to `MOMASConfig` |
| `experiments/configs/*.yaml` | Add `welfare_function` and `utility_function_type` where needed |
| `tests/` | New tests for the three changes |
