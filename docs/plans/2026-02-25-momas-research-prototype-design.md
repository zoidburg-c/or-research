# MOMAS Research Prototype Design

**Date:** 2026-02-25
**Based on:** Radulescu et al. (2019) — Multi-Objective Multi-Agent Decision Making (arXiv:1909.02964)

## Goal

Build working RL simulation prototypes applying the MOMAS framework to two domains:
1. **Algorithmic trading** — multi-agent order book
2. **Healthcare resource allocation** — hospital bed/ICU allocation

Run all 5 taxonomy settings from the paper across both domains (10 experimental configurations), evaluate under both SER and ESR criteria.

## Tech Stack

- Python 3.11+
- PettingZoo — multi-agent environment API
- MO-Gymnasium / MORL-Baselines — multi-objective RL
- NumPy, PyTorch — numerical compute and deep RL
- Matplotlib/Seaborn — visualization

## Project Structure

```
or-research/
├── momas/                  # Core MOMAS framework
│   ├── __init__.py
│   ├── base_env.py         # PettingZoo-compatible MOMAS wrapper
│   ├── utility.py          # Utility and social welfare functions
│   └── metrics.py          # Pareto fronts, hypervolume, coverage sets
├── envs/
│   ├── __init__.py
│   ├── trading/            # Order book multi-agent market
│   │   ├── __init__.py
│   │   ├── orderbook_env.py
│   │   ├── orderbook.py    # LOB matching engine
│   │   └── agents.py       # Agent type definitions (MM, momentum, arb)
│   └── healthcare/         # Hospital bed allocation
│       ├── __init__.py
│       ├── hospital_env.py
│       ├── patients.py     # Patient generation and dynamics
│       └── hospital.py     # Hospital state management
├── agents/                 # Shared MORL agent implementations
│   ├── __init__.py
│   ├── mo_q_learning.py    # Tabular baseline
│   ├── pareto_dqn.py       # Deep Pareto Q-learning
│   ├── mo_ppo.py           # Multi-policy MO-PPO
│   └── envelope_moq.py     # Envelope MOQ-Learning
├── experiments/            # Experiment runner and configs
│   ├── __init__.py
│   ├── runner.py           # Config-driven experiment runner
│   └── configs/            # YAML experiment configs
├── results/                # Experiment outputs (gitignored)
├── notebooks/              # Analysis and visualization
├── tests/                  # Unit tests
├── docs/
│   └── plans/
├── plan.md                 # Original survey notes
├── pyproject.toml
└── README.md
```

---

## 1. Core MOMAS Framework (`momas/`)

### 1.1 MOMAS Config and Environment Wrapper (`base_env.py`)

Wraps any PettingZoo `ParallelEnv` and applies the taxonomy configuration.

```python
@dataclass
class MOMASConfig:
    reward_structure: Literal["team", "individual"]
    utility_type: Literal["team", "social_choice", "individual"]
    optimisation_criterion: Literal["SER", "ESR"]
    num_objectives: int
    utility_functions: dict[str, Callable]  # per-agent or shared
```

Responsibilities:
- **Team reward mode:** All agents receive the same vector reward (copied from the environment's global reward)
- **Individual reward mode:** Each agent gets its own reward vector from the environment
- **SER evaluation:** Apply utility to the expected return vector across episodes
- **ESR evaluation:** Apply utility per episode, then average the scalar utilities
- The wrapper transforms rewards and evaluation — it does not change underlying environment dynamics

### 1.2 Utility Functions (`utility.py`)

```python
# Linear: weighted sum
linear_utility(weights: np.ndarray) -> Callable

# Threshold: minimum requirement per objective, then weighted sum of surplus
threshold_utility(thresholds: np.ndarray, weights: np.ndarray) -> Callable

# Chebyshev: minimize worst-case weighted deviation from ideal point
chebyshev_utility(weights: np.ndarray, ideal: np.ndarray) -> Callable

# Social welfare aggregators
utilitarian_welfare(individual_utilities: list[float]) -> float  # sum
nash_welfare(individual_utilities: list[float]) -> float         # product
lorenz_welfare(individual_utilities: list[float]) -> float       # fairness-oriented
```

### 1.3 Evaluation Metrics (`metrics.py`)

- **Pareto front computation:** Given a set of policy return vectors, compute the Pareto-undominated set
- **Hypervolume indicator:** Standard metric for multi-objective solution set quality
- **Coverage set analysis:** Check whether a solution set covers all utility functions in a given class
- **SER vs ESR evaluation:** Evaluate policies under both criteria for direct comparison

---

## 2. Trading Environment (`envs/trading/`)

### 2.1 Order Book (`orderbook.py`)

A simplified limit order book with price-time priority matching:
- Discrete tick-based simulation
- Orders that cross the spread execute immediately
- Unmatched orders sit in the book at their price level
- Configurable tick size and book depth

### 2.2 Environment (`orderbook_env.py`)

PettingZoo `ParallelEnv` — multiple agents submit orders and their actions move prices.

**State space** (per agent, partial observability):
- Current mid-price, bid-ask spread
- Agent's own inventory position and cash
- Recent price history (rolling window, e.g., last 50 ticks)
- Agent's own open orders
- Public order book depth (top N levels)

**Action space** (discrete, per agent):
- Place limit buy/sell at various price levels relative to mid
- Place market buy/sell
- Cancel existing orders
- Hold (no action)

**Objectives (d=3):**
1. **P&L** — realized + unrealized profit
2. **Risk** — negative of drawdown or portfolio variance (higher = less risk)
3. **Liquidity cost** — negative of slippage/market impact (higher = lower cost)

**Agent types** (configurable mix):
- **Market makers** — profit from spread, want low inventory risk
- **Momentum traders** — follow trends, want high returns
- **Arbitrageurs** — exploit mispricings, want low-risk profit

**Config:**
```python
trading_config = dict(
    num_agents=4,
    agent_types=["market_maker", "market_maker", "momentum", "arbitrageur"],
    num_objectives=3,
    orderbook_depth=10,
    tick_size=0.01,
    initial_cash=100_000,
    episode_length=1000,
)
```

### 2.3 Taxonomy Mapping

| Setting | Trading Interpretation |
|---------|----------------------|
| Team reward, team utility | Co-operative trading desk: all agents share total desk P&L, same risk mandate |
| Team reward, social choice | Shared P&L but a portfolio manager aggregates agents' different risk preferences via welfare function |
| Team reward, individual utility | Shared P&L but each trader values risk/return differently (aggressive vs conservative) |
| Individual reward, social choice | Competing firms but a regulator optimizes market-level welfare (spread, volatility, fairness) |
| Individual reward, individual utility | Fully competitive: each agent maximizes own P&L/risk/cost trade-off |

---

## 3. Healthcare Environment (`envs/healthcare/`)

### 3.1 Patient Generation (`patients.py`)

- Stochastic arrivals with configurable Poisson distributions
- Multiple acuity levels: critical (needs ICU), acute (needs general bed), low-acuity (can wait)
- Each patient has a time-sensitivity: outcomes degrade if not treated within a window
- Configurable surge patterns (time-of-day, seasonal)

### 3.2 Hospital State (`hospital.py`)

- Bed inventory by type (general, ICU, emergency)
- Staffing levels affecting treatment capacity
- Patient treatment progression (stochastic length-of-stay)

### 3.3 Environment (`hospital_env.py`)

PettingZoo `ParallelEnv` — multiple hospitals deciding how to allocate limited beds.

**State space** (per hospital agent):
- Current bed occupancy by type (general, ICU, emergency)
- Queue of incoming patients with acuity scores and estimated length-of-stay
- Current staffing levels
- Neighbouring hospitals' publicly reported occupancy (partial observability)
- Time of day / day of week (affects arrival rates)

**Action space** (discrete, per hospital):
- Accept patient to available bed type
- Divert patient to neighbouring hospital
- Discharge a stable patient early to free capacity
- Request transfer of a patient to/from another hospital

**Objectives (d=3):**
1. **Mortality reduction** — proportion of patients receiving timely appropriate care
2. **Equity/fairness** — evenness of wait times and outcomes across patient demographics
3. **Cost efficiency** — resource utilization rate (avoid both waste and overload)

**Dynamics:**
- Patients in beds progress through treatment (stochastic length-of-stay)
- Diversions and transfers have a time cost and risk penalty
- Arrival rates can spike (simulating surges/seasonal patterns)

**Config:**
```python
healthcare_config = dict(
    num_hospitals=3,
    beds_per_hospital={"general": 50, "icu": 10, "emergency": 5},
    num_objectives=3,
    patient_arrival_rate=2.0,
    acuity_distribution=[0.6, 0.3, 0.1],
    episode_length=168,
)
```

### 3.4 Taxonomy Mapping

| Setting | Healthcare Interpretation |
|---------|--------------------------|
| Team reward, team utility | Regional health system: all hospitals share region-wide outcome metrics, same mandate |
| Team reward, social choice | Shared regional outcomes but a health authority balances hospitals' differing priorities (rural vs urban) |
| Team reward, individual utility | Regional metrics shared but each hospital weights mortality vs cost differently |
| Individual reward, social choice | Each hospital measured independently but a regulator optimizes regional welfare |
| Individual reward, individual utility | Fully independent hospitals competing for patients and funding |

---

## 4. Agents & Training (`agents/`, `experiments/`)

### 4.1 Agent Implementations

Use MORL-Baselines where possible, wrapping for multi-agent compatibility:

| Agent | Type | Use Case |
|-------|------|----------|
| Multi-Objective Q-Learning | Tabular | Baseline for small state spaces |
| Pareto DQN | Deep RL | Maintains Pareto-optimal Q-value vectors |
| Multi-Policy MO-PPO | Policy gradient | Discovers multiple Pareto-optimal policies |
| Envelope MOQ-Learning | Value-based | Optimistic linear scalarization for CCS discovery |

Each agent wraps to PettingZoo's API. The MOMAS wrapper handles reward transformation; agents see vector rewards and optimize.

### 4.2 Experiment Runner (`experiments/runner.py`)

```python
@dataclass
class ExperimentConfig:
    env: str                    # "trading" or "healthcare"
    env_params: dict
    momas_config: MOMASConfig
    agent_type: str
    num_episodes: int
    seed: int
```

Pipeline per experiment:
1. Instantiate the domain environment
2. Wrap with MOMAS config
3. Create agents
4. Train for N episodes
5. Evaluate under both SER and ESR
6. Compute Pareto fronts and hypervolume
7. Save results to structured output directory

### 4.3 Full Sweep

10 configurations (5 taxonomy settings x 2 domains) x multiple agent types x multiple seeds.

```
results/
├── trading/
│   ├── team_reward-team_utility/
│   ├── team_reward-social_choice/
│   ├── team_reward-individual_utility/
│   ├── individual_reward-social_choice/
│   └── individual_reward-individual_utility/
└── healthcare/
    └── (same structure)
```

---

## 5. Analysis & Visualization (`notebooks/`)

Jupyter notebooks for:
- **Pareto front plots** — per setting, comparing agent types
- **Hypervolume convergence** — learning curves showing multi-objective performance over training
- **SER vs ESR comparison** — how the optimisation criterion changes which policies are preferred
- **Cross-domain comparison** — does the same taxonomy setting behave similarly across trading and healthcare?
- **Utility sensitivity analysis** — how do different utility function shapes (linear, threshold, Chebyshev) change the optimal policies?
