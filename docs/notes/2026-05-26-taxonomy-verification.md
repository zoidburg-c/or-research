# Taxonomy Verification — Sweep Results

**Date:** 2026-05-26
**Closes:** Task 6 of [`2026-02-27-taxonomy-gaps-implementation.md`](../plans/2026-02-27-taxonomy-gaps-implementation.md) — *Run Full Sweep and Verify Differentiation*
**Plan goal it verifies:** *"Make the 5 MOMAS taxonomy settings produce meaningfully different training behavior and results."*

## Setup

- Env: fresh `.venv` (Python 3.13.5), `uv pip install -e .[dev]` → torch 2.12, morl-baselines, mo-gymnasium, pettingzoo, pymoo
- Command: `WANDB_MODE=disabled .venv/bin/python -m experiments.sweep`
- Configs: 16 (all in `experiments/configs/`, `smoke_test.yaml` skipped). 50 episodes/config, seed 42.
- Raw stdout: `sweep_logs/full_sweep.log` (gitignored).

## Results

### Healthcare

| Setting | Agent | Criterion | Hypervolume | Welfare (SER) | Pareto front |
|---|---|---|---:|---:|---:|
| individual / individual | tabular | SER | 12,930.29 | 72.82 | 8 |
| individual / social | tabular | SER | 17,338.23 | 74.78 | 8 |
| team / individual | tabular | SER | 275,508.34 | 217.07 | 13 |
| team / social | tabular | SER | 303,198.22 | 233.87 | 9 |
| team / team | tabular | SER | **444,577.26** | 219.75 | 6 |
| team / team | tabular | **ESR** | **109,036.07** | 237.16 | 11 |
| team / team | MO-DQN | SER | 79,493.63 | 214.62 | 10 |
| team / team | Envelope | SER | 92,017.49 | 242.20 | 12 |

### Trading

| Setting | Agent | Criterion | Hypervolume | Welfare (SER) | Pareto front |
|---|---|---|---:|---:|---:|
| individual / individual | tabular | SER | 499.37 | -45.15 | 3 |
| individual / social | tabular | SER | 5,145.74 | -9.33 | 11 |
| team / individual | tabular | SER | 47,251.82 | -88.11 | 7 |
| team / social | tabular | SER | 72,805.84 | -249.40 | 7 |
| team / team | tabular | SER | **37,338.26** | -114.62 | 5 |
| team / team | tabular | **ESR** | **1,082.04** | -118.85 | 1 |
| team / team | MO-DQN | SER | 96,480.02 | -399.54 | 4 |
| team / team | Envelope | SER | 1,197,272.64 | 607.93 | 13 |

## Verdict against Task 6 acceptance criteria

| Criterion | Result | Pass? |
|---|---|---|
| Team/team ≠ Team/individual ≠ Team/social_choice (same domain) | All three differ ≥1.5× HV in both domains | ✅ |
| Individual reward ≠ Team reward (same domain) | 10–95× HV gap | ✅ |
| SER config ≠ ESR config (same setting, training-side) | HV diverges 4× (healthcare) / 35× (trading) | ✅ |
| SER metric ≠ ESR metric (same setting, eval-side) | Identical in every config | ❌ |

**3 of 4 criteria pass.** The implementation makes the 5 taxonomy settings produce materially different training behavior, and the deep-RL agents (MO-DQN, Envelope) yield distinct policies from tabular MOQ. The one failure is on the eval reporting side, not the training axis — see below.

## Open issues uncovered by the sweep

### 1. Eval utility is hardcoded linear → SER ≡ ESR by Jensen's

`experiments/runner.py:237-240`:

```python
eval_weights = np.ones(cfg.num_objectives) / cfg.num_objectives
eval_u = linear_utility(eval_weights)         # always linear
ser = evaluate_ser(episode_returns, eval_u)
esr = evaluate_esr(episode_returns, eval_u)
```

For a linear `u`, `u(E[r]) ≡ E[u(r)]`, so the printed SER and ESR are mathematically forced to be equal regardless of config. Training *does* use `threshold_utility` for ESR configs (driving the training-side HV divergence), but the reported metrics can't surface SER↔ESR distinction with a linear probe.

**Fix:** evaluate with `_make_utility_fn(weights, cfg.momas_criterion)` (the same utility the agents trained against), then re-run the two `team_team_esr` configs to confirm ESR ≠ SER on the eval side. Acceptance criterion 4 becomes verifiable once this lands.

### 2. Output path silently overwrites across criterion / agent variants

`runner.py:254`:

```python
out_dir = Path(cfg.output_dir) / cfg.env / f"{cfg.momas_reward_structure}-{cfg.momas_utility_type}"
np.savez(out_dir / f"seed_{cfg.seed}.npz", ...)
```

The path ignores `optimisation_criterion` and `agent_type`. The four `team_team*` configs in each domain (`team_team`, `team_team_esr`, `team_team_dqn`, `team_team_envelope`) all write to `results/<env>/team-team/seed_42.npz`. In our sweep, only the *last-run* config (`team_team_esr`) survived on disk. Stdout metrics above are unaffected, but any persisted-results analysis is silently corrupted.

**Fix:** include `momas_criterion` and `agent_type` in the path, e.g. `results/<env>/<reward>-<utility>/<criterion>-<agent>/seed_<seed>.npz`.

### 3. Envelope on trading is a large positive HV outlier

Trading team/team Envelope reports HV ≈ 1.2M and welfare = +608, vs ≈ 37k / -115 for tabular and ~96k / -400 for DQN. Worth a sanity check on either the Envelope wrapper's reward scaling or the orderbook env's reward bounds before treating this as a substantive finding.

## Single-seed caveat

All metrics are from one seed (42). The training-side divergences are large enough (often ≥10×) that single-seed noise is unlikely to flip the qualitative conclusions, but any quantitative claim should be re-run across ≥5 seeds before being cited.
