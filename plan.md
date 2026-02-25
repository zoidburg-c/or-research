# Multi-Objective Multi-Agent Decision Making: A Utility-based Analysis and Survey

**Paper:** [arXiv:1909.02964](https://arxiv.org/abs/1909.02964)
**Authors:** Roxana Rădulescu, Patrick Mannion, Diederik M. Roijers, Ann Nowé (2019)

## Problem Statement

Most multi-agent system (MAS) implementations optimize agents' policies with respect to a single objective, despite the fact that many real-world problems are inherently multi-objective. Multi-objective multi-agent systems (MOMAS) explicitly consider trade-offs between conflicting objective functions. The paper argues that compromises between objectives should be analyzed on the basis of the utility they provide to users.

## Core Concepts

### Utility-Based Approach

- User preferences are modeled via **utility functions** that map vector-valued returns to scalar values.
- The ultimate goal is to maximize **user utility**, and what constitutes a solution should be derived from what is known about user utility.

### Two Optimisation Criteria

| Criterion | Definition | When to Use |
|-----------|-----------|-------------|
| **SER** (Scalarised Expected Returns) | Apply utility function to the *expected* return vector: `V = u(E[Σ γ^t r_t])` | When evaluating a policy over many executions (average performance matters) |
| **ESR** (Expected Scalarised Returns) | Apply utility function to *each individual* return, then take expectation: `V = E[u(Σ γ^t r_t)]` | When the outcome of a single execution matters |

- For **linear** utility functions, SER and ESR are equivalent.
- For **non-linear** utility functions, they may yield different optimal policies.

### Utility Functions

- **Linear:** Weighted sum of objectives — `u(r) = Σ w_d * r_d`
- **Non-linear / discontinuous:** E.g., threshold-based functions where a minimum payoff on an objective is required.
- **Monotonically increasing:** The general class of interest — more of each objective is always preferred.

## Formal Model: MOPOSG

The most general model is the **Multi-Objective Partially Observable Stochastic Game (MOPOSG)**, defined as a tuple `M = (S, A, T, R)` with `n ≥ 2` agents and `d ≥ 2` objectives:

- `S` — state space
- `A = A_1 × ... × A_n` — joint action set
- `T: S × A × S → [0,1]` — probabilistic transition function
- `R = R_1 × ... × R_n` — vectorial reward functions, `R_i: S × A × S → R^d`

### Special Case Models

Many existing models are derived by restricting MOPOSG parameters:

| Model | Objectives (d) | Agents (n) | States (\|S\|) | Observability |
|-------|---------------|------------|----------------|---------------|
| MOPOSG | — | — | — | — |
| MOSG | — | — | — | full |
| MODec-POMDP | — | — | — | — |
| MOMMDP | — | — | — | full |
| MOCoG | — | 1 (per node) | — | full |
| MONFG | 2 | 1 | — | full |
| MOMDP | 1 agent | — | — | full |
| MO Multi-armed bandit | 1 agent | 1 | 1 | full |

See paper Figure 3 for the full Venn diagram of model relationships.

## Taxonomy of Settings

The paper classifies MOMAS problems along two axes:

### Axis 1: Reward Structure

- **Team reward:** All agents receive the same reward vector (`R_1 = ... = R_n = R`)
- **Individual rewards:** Each agent receives a different reward/return vector

### Axis 2: Utility Type

- **Team utility:** All agents share a single utility function
- **Social choice utility:** A social welfare function aggregates individual agent utilities
- **Individual utility:** Each agent has its own utility function and optimizes for itself

### Resulting Settings (5 combinations)

```
Multi-Objective Multi-Agent Decision Making
├── Team Reward
│   ├── Team Utility          (fully cooperative, reducible to single-agent MOMDP)
│   ├── Social Choice         (optimize overall social welfare)
│   └── Individual Utility    (shared rewards, but agents value them differently)
└── Individual Reward
    ├── Social Choice         (optimize social welfare with different reward vectors)
    └── Individual Utility    (fully self-interested agents)
```

**Note:** Individual rewards with team utility is not a meaningful combination — even identical utility functions produce different scalar values from different reward vectors.

## Use-Case Scenarios for Multi-Objective Decision Making

1. **Unknown weights / unknown utility function:** Utility is uncertain or unknown at planning time. Compute a **coverage set** — a set of solutions containing at least one optimal policy for every possible utility function.
2. **Decision support:** Utility function cannot be explicitly specified (e.g., medical treatment decisions). Compute a coverage set and present alternatives to the user.
3. **Known weights / known utility function:** Utility is known a priori, but a priori scalarisation may still be intractable for non-linear functions.

## Solution Concepts

### Coverage Sets

- **Pareto coverage set (PCS):** Set of all Pareto-undominated policies (no other policy is better on all objectives).
- **Convex coverage set (CCS):** Coverage set with respect to all possible linear utility functions.
- For stochastic policies, a Pareto coverage set can be constructed from a CCS of deterministic stationary policies.

### Game-Theoretic Solutions (for individual utilities)

- **Nash equilibria:** No agent has an incentive to unilaterally deviate.
- **Core stability:** No coalition of agents can improve by deviating together.
- **Negotiation:** Agents agree on a joint policy, potentially selecting non-stable solutions that offer better utility for all.

### Social Welfare Concepts

- **Mechanism design:** Construct a payment system so agents truthfully report preferences, making the problem fully cooperative.
- **Lorenz optimality:** Fairness-oriented — prefer policies where values are more evenly distributed across agents.

## Key Findings and Gaps

### Well-Studied Areas

- Team reward with team utility under SER (the simplest fully cooperative case)
- Multi-objective coordination graphs (MOCoG) — one of the most well-studied models

### Under-Explored Areas

- **ESR criterion:** Rarely studied in multi-agent settings, despite being important when single-execution outcomes matter.
- **Individual utilities with team rewards:** Complex coordination is needed even though rewards are shared.
- **Social choice in multi-objective settings:** Mechanism design and social welfare functions with vector-valued returns.
- **Individual rewards with individual utilities:** The fully self-interested multi-objective case.
- **Non-linear utility functions:** Most existing work assumes linear scalarisation.

## Challenges in Multi-Agent Multi-Objective Settings

1. **Exponential joint action space:** Number of possible joint actions grows exponentially with the number of agents, leading to much larger coverage sets.
2. **Non-stationarity:** Each agent's learning is affected by other agents' changing policies.
3. **Coordination:** Even with shared rewards, agents with different utilities must coordinate on joint policies.
4. **Unknown utility functions:** When agents cannot or will not reveal their preferences, learning stable solutions requires discovering individual utility functions.
5. **Loose couplings:** Exploiting factored reward structures (coordination graphs) is key to tractability.

## Future Research Directions

- Extending single-agent multi-objective methods to multi-agent settings
- Developing ESR-specific algorithms for multi-agent problems
- Applying mechanism design to multi-objective multi-agent settings
- Studying fairness (Lorenz optimality) in multi-objective contexts
- Scaling deep reinforcement learning approaches to multi-objective multi-agent problems
- Bridging game theory and multi-objective optimization more tightly
