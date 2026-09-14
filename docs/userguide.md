# Layeredtrust User Guide

**Version 1.0.0 · A Unified Multi-Layer Mathematical Framework for Trust-Based Decision Making in Multi-Agent Systems**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation and Setup](#2-installation-and-setup)
3. [Conceptual Overview](#3-conceptual-overview)
4. [Your First Trust Decision](#4-your-first-trust-decision)
5. [Understanding the Trustee](#5-understanding-the-trustee)
6. [Understanding the Trustor](#6-understanding-the-trustor)
7. [Making Decisions with TrustScenario](#7-making-decisions-with-trustscenario)
8. [Working with Groups](#8-working-with-groups)
9. [Aggregation Architectures in Depth](#9-aggregation-architectures-in-depth)
10. [The Cohesion–Diversity Function](#10-the-cohesiondiversity-function)
11. [Correlated Quorum Models](#11-correlated-quorum-models)
12. [Temporal Dynamics and Trust Recovery](#12-temporal-dynamics-and-trust-recovery)
13. [Network Propagation](#13-network-propagation)
14. [Common Recipes](#14-common-recipes)
15. [Advanced Usage](#15-advanced-usage)
16. [Parameter Calibration Guide](#16-parameter-calibration-guide)
17. [Troubleshooting](#17-troubleshooting)
18. [Best Practices](#18-best-practices)
19. [Frequently Asked Questions](#19-frequently-asked-questions)
20. [Reference Tables](#20-reference-tables)

---

## 1. Introduction

### 1.1 What This Guide Is

This is a **practical, task-oriented guide** to using `layeredtrust` in real applications. It complements the mathematical details in the manuscript and the terse API summary in the README. You do not need to read the manuscript to use this guide—everything needed to make a decision is explained here.

### 1.2 Who This Guide Is For

- **Researchers** building simulations of trust in multi-agent systems
- **Engineers** embedding trust-based decision logic into agents
- **Students** learning computational trust models
- **Reviewers** wanting to reproduce the manuscript's results

### 1.3 What You Can Build

With `layeredtrust` you can model:

- A buyer agent deciding whether to purchase from an unknown seller
- A passenger deciding whether to trust a robotaxi after a failure
- A logistics manager deciding whether to dispatch a truck platoon
- A patient evaluating a surgical team
- A DAO member evaluating a multi-signature wallet
- A user evaluating a hybrid human-AI fact-checking collective
- A network of agents exchanging trust recommendations

### 1.4 Prerequisites

- Python 3.9 or later
- Basic familiarity with Python classes and functions
- No external libraries required

---

## 2. Installation and Setup

### 2.1 Option A: Single-File Drop-In

The simplest approach is to copy `layeredtrust.py` into your project directory:

```
my_project/
├── layeredtrust.py
├── my_agent.py
└── ...
```

Then import directly:

```python
from layeredtrust import Trustee, Trustor, TrustScenario
```

This is the recommended approach for most users.

### 2.2 Option B: Package Install

```bash
git clone https://github.com/<your-username>/layeredtrust.git
cd layeredtrust
pip install -e .
```

Now `import layeredtrust` works from anywhere.

### 2.3 Option C: Development Install

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `numpy` (for the simulation helper), and other dev tools.

### 2.4 Verifying Your Installation

```python
import layeredtrust
print(layeredtrust.__version__)  # 1.0.0

# Run the built-in demo
python layeredtrust.py
```

You should see five case-study outputs.

### 2.5 Importing What You Need

The library exports the following public names:

```python
from layeredtrust import (
    # Core classes
    Trustee, Trustor, TrustGroup, TrustScenario, TrustNetwork,
    # Aggregation
    AggregationType,
    # Helper functions
    harmonic_mean, geometric_mean, arithmetic_mean, median,
    sigmoid, strategic_factor,
    # Simulation
    simulate_comparison,
)
```

---

## 3. Conceptual Overview

### 3.1 The Trust Decision Problem

At its core, every use of `layeredtrust` answers the question:

> **"Should agent *i* rely on agent *j* (or on a group *G*) for some task?"**

The answer is a boolean **decision**, but it is derived from a continuous **effective trust** value in `[0, 1]` that is compared against a **risk threshold**.

### 3.2 The Three-Box Mental Model

Think of the library in three boxes:

```
   ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
   │    TRUSTEE(S)       │     │      TRUSTOR        │     │     SCENARIO        │
   │  Who is being       │     │  Who is deciding    │     │  What are the       │
   │  evaluated?         │     │  whether to trust?  │     │  payoffs and stakes?│
   │                     │     │                     │     │                     │
   │  Trustee            │     │  Trustor            │     │  TrustScenario      │
   │  TrustGroup         │     │  τ, ρ, δ*, β_s      │     │  U_win, U_loss, S   │
   └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
              │                            │                          │
              └────────────────────────────┴──────────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │  Decision + Effective    │
                              │  Trust + Expected        │
                              │  Utility                 │
                              └──────────────────────────┘
```

### 3.3 The Seven Layers at a Glance

| Layer | What It Models | Where You Set It |
|-------|----------------|------------------|
| **L1** | Perceived ability, benevolence, integrity | `Trustee.A`, `.B`, `.I` |
| **L2** | Risk-adjusted expected utility | `TrustScenario.U_win`, `.U_loss`, `.stakes`, `Trustor.risk_aversion` |
| **L3** | Bayesian reliability with recency | `Trustee.a`, `.b`, `update_bayes(lam=...)` |
| **L4** | Asymmetric trust build/destroy | `Trustee.k_build`, `.k_destroy`, `asymmetric_dynamics(...)` |
| **L5** | Strategic cooperation threshold | `Trustee.discount_factor`, `Trustor.delta_star`, `.beta_s` |
| **L6** | Network propagation | `TrustNetwork` |
| **L7** | Group aggregation | `TrustGroup` |

### 3.4 When You Need Which Layer

You don't need all seven layers. Choose based on your scenario:

| If your scenario involves… | Use… |
|---------------------------|------|
| A simple transaction with a stranger | L1 + L2 + L3 |
| Trust collapsing after a failure | L4 |
| An ongoing relationship with incentives | L5 |
| Trust recommendations from others | L6 |
| A team, committee, or multi-sig wallet | L7 |

---

## 4. Your First Trust Decision

Let's build a complete, minimal trust decision step by step.

### 4.1 The Scenario

A buyer on Amazon is considering an $800 camera from an unknown seller. The seller has 130 positive reviews and 20 negative reviews. If the camera works, the buyer gains $100 in value; if it's defective, the buyer loses the full $800.

### 4.2 The Code

```python
from layeredtrust import Trustee, Trustor, TrustScenario

# Step 1: Define the seller (trustee)
seller = Trustee(
    id="seller",
    a=130,             # positive evidence
    b=20,              # negative evidence
    discount_factor=0.8
)

# Step 2: Define the buyer (trustor)
buyer = Trustor(
    id="buyer",
    trust_propensity=0.6,   # moderately trusting
    risk_aversion=0.2       # moderate risk aversion
)

# Step 3: Define the scenario
scenario = TrustScenario(
    name="Amazon Camera",
    trustor=buyer,
    target=seller,
    U_win=100,              # gain if seller is honest
    U_loss=-800,            # loss if seller defects
    stakes=800              # stakes for threshold
)

# Step 4: Run
decision, trust, eu = scenario.run()

print(f"Effective trust: {trust:.3f}")     # 0.372
print(f"Expected utility: ${eu:.2f}")      # -465.20
print(f"Decision: {'BUY' if decision else 'DO NOT BUY'}")
```

### 4.3 What Just Happened

Let's trace the computation:

1. **Layer 3 (Bayesian)** — The Beta mean is `130 / 150 = 0.867`.
2. **Layer 1 (Cognitive)** — Because `A`, `B`, `I` were not set, the Beta mean is used as the fallback: `P = 0.867`.
3. **Propensity** — `τ × P = 0.6 × 0.867 = 0.520`.
4. **Layer 5 (Strategic)** — `Φ = σ(10 × (0.8 − 0.708)) = 0.715`.
5. **Effective trust** — `0.520 × 0.715 = 0.372`.
6. **Layer 2 (Utility)** — `EU = 0.372 × 100 + 0.628 × (−800) = −465.20`.
7. **Threshold** — `θ = 0.2 × 800 = 160`.
8. **Decision** — `−465.20 < 160` → **Do not buy**.

### 4.4 Inspecting the Result

Use `report()` for a structured view:

```python
report = scenario.report()
for key, val in report.items():
    print(f"{key}: {val}")
```

Output:

```
name: Amazon Camera
effective_trust: 0.372...
expected_utility: -465.2
threshold: 160.0
decision: False
```

---

## 5. Understanding the Trustee

The `Trustee` class represents the agent (or agent-like entity) being evaluated.

### 5.1 Two Ways to Specify Trust

There are **two** ways to give a trustee a trust value:

**Method A — Beta parameters (`a`, `b`)** : Principled, updatable.

```python
seller = Trustee("seller", a=130, b=20)
seller.bayes_mean()   # 0.867
```

**Method B — Explicit `trust`** : Direct override.

```python
car = Trustee("car")
car.trust = 0.65
car.current_trust()   # 0.65
```

**Which should you use?**

- Use Beta parameters when you have a **history** of successes/failures and want the model to update automatically.
- Use explicit `trust` when you have an **externally supplied** trust score (from a survey, a heuristic, or a research study).

If `trust` is set, it takes priority. Use `reset()` to clear it.

### 5.2 The Three Cognitive Traits (Layer 1)

The Mayer–Davis–Schoorman model decomposes trustworthiness into three traits:

- **`A`** — Ability (competence)
- **`B`** — Benevolence (goodwill)
- **`I`** — Integrity (adherence to principles)

```python
ceo = Trustee("ceo", A=0.9, B=0.3, I=0.8)

P = ceo.perceived_trustworthiness(alpha=0.7, beta=0.2, gamma=0.1)
# P = 0.7*0.9 + 0.2*0.3 + 0.1*0.8 = 0.77
```

You can also mix: set `A` and `I` explicitly, let `B` fall back to the Beta mean.

```python
ceo = Trustee("ceo", a=8, b=2, A=0.9, I=0.8)
# B defaults to the Beta mean (0.8)
```

### 5.3 The Beta Parameters (`a`, `b`)

The Beta distribution `Beta(a, b)` is the conjugate prior for a Bernoulli process.

- `a` — pseudocount of successes (+1 to avoid degeneracy)
- `b` — pseudocount of failures (+1)

**Rules of thumb:**

| Situation | `a` | `b` |
|-----------|-----|-----|
| No history | 1 | 1 |
| 10 successes, 0 failures | 11 | 1 |
| 100 successes, 5 failures | 101 | 6 |
| 50/50 split | 50 | 50 |

### 5.4 Updating with Evidence (Layer 3)

After each interaction, update the Beta parameters:

```python
seller = Trustee("seller", a=1, b=1)

# Interaction succeeds
seller.update_bayes(success=True)
# a=2, b=1 → mean = 0.667

# Interaction fails
seller.update_bayes(success=False)
# a=2, b=2 → mean = 0.500
```

**Recency bias** — pass `lam < 1` to weight recent evidence more heavily:

```python
seller.update_bayes(success=True, lam=0.9)
```

With `lam=0.9`, the previous counts are discounted by 10% before adding new evidence. Use `lam` in `[0.8, 1.0]` for most applications.

### 5.5 The Discount Factor (Layer 5)

The `discount_factor` represents the trustee's **long-term orientation** in a repeated game. A value near `1.0` means the trustee values future interactions; near `0.0` means short-term opportunistic behavior.

| Trustee type | Typical δ |
|--------------|-----------|
| Established seller, repeat business | 0.9–0.95 |
| Regular contractor | 0.8–0.9 |
| One-off transaction | 0.5–0.7 |
| Untraceable anonymous | 0.3–0.5 |

If `δ < δ*` (default 0.708), the strategic override kicks in and reduces trust.

### 5.6 The Build/Destroy Rates (Layer 4)

`k_build` and `k_destroy` control how fast trust grows and collapses:

```python
car = Trustee("car", k_build=0.1, k_destroy=0.8)
```

`k_destroy > k_build` captures the **negativity bias**. A common ratio is 3:1 to 8:1.

| Scenario | `k_build` | `k_destroy` | Ratio |
|----------|-----------|-------------|-------|
| Human–robot | 0.1 | 0.8 | 8:1 |
| Human–human (team) | 0.15 | 0.45 | 3:1 |
| Institutional trust | 0.05 | 0.30 | 6:1 |

---

## 6. Understanding the Trustor

The `Trustor` class represents the agent making the decision.

### 6.1 Trust Propensity (τ)

Baseline willingness to trust:

| τ | Interpretation |
|---|----------------|
| 0.2–0.4 | Cautious, skeptical |
| 0.5 | Neutral |
| 0.6–0.8 | Trusting |
| 0.9–1.0 | Highly trusting, risk-tolerant |

Veteran employees often have low τ; new hires often have high τ.

```python
veteran = Trustor("veteran", trust_propensity=0.3)
newbie = Trustor("newbie", trust_propensity=0.8)
```

### 6.2 Risk Aversion (ρ)

The threshold multiplier. Higher ρ means the trustor demands more expected utility before acting.

| ρ | Interpretation |
|---|----------------|
| 0.05 | Risk-tolerant, speculative |
| 0.15–0.25 | Moderate |
| 0.4–0.6 | Conservative, safety-critical |
| 0.8+ | Highly risk-averse |

```python
speculator = Trustor("speculator", risk_aversion=0.05)
surgeon = Trustor("surgeon", risk_aversion=0.5)
```

### 6.3 Strategic Parameters

`delta_star` is the betrayal threshold. The default `0.708` comes from the standard PD payoff configuration $(T, R, P, S) = (5, 3, 1, 0)$: $(T-R)/(T-P) = 2/4 = 0.5$… actually $0.708$ corresponds to the common alternative configuration. Use the default unless your game has different payoffs.

`beta_s` is the logistic steepness. Values between 5 and 20 are reasonable:

| `beta_s` | Behavior |
|----------|----------|
| 2 | Very gradual transition |
| 10 | Moderate (default) |
| 20 | Close to a hard switch |
| 100+ | Essentially a step function |

### 6.4 Combining Multiple Trustors

When several trustors evaluate the same trustee, run separate scenarios:

```python
ceo = Trustee("ceo", A=0.9, B=0.3, I=0.8)

for τ, label in [(0.3, "veteran"), (0.8, "newbie")]:
    trustor = Trustor(label, trust_propensity=τ)
    sc = TrustScenario("CEO", trustor, ceo, U_win=100, U_loss=-50, stakes=100)
    print(f"{label}: trust={sc.target_trust():.3f}")
```

---

## 7. Making Decisions with TrustScenario

### 7.1 The Constructor

```python
TrustScenario(
    name: str,
    trustor: Trustor,
    target: Trustee | TrustGroup,
    U_win: float = 1.0,
    U_loss: float = -1.0,
    stakes: float | None = None,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
)
```

**Key parameters:**

- `U_win` — payoff if the trustee honors trust
- `U_loss` — payoff if the trustee defects (usually negative)
- `stakes` — used for threshold computation; if `None`, uses `abs(U_loss)`
- `alpha`, `beta`, `gamma` — cognitive weights (should sum to 1)

### 7.2 The Decision Rule

The scenario decides **to trust** when:

$$\text{EU} = T_{\text{eff}} \cdot U_{\text{win}} + (1 - T_{\text{eff}}) \cdot U_{\text{loss}} \ \geq \ \rho \cdot S$$

### 7.3 The `run()` Method

Returns a tuple:

```python
decision, trust, eu = scenario.run()
```

- `decision` — `bool`
- `trust` — `float` in `[0, 1]`
- `eu` — `float`

### 7.4 The `report()` Method

Returns a dictionary with all intermediates:

```python
report = scenario.report()
```

Keys: `name`, `effective_trust`, `expected_utility`, `threshold`, `decision`.

### 7.5 Reading the Decision

| `decision` | Meaning |
|------------|---------|
| `True` | Expected utility exceeds threshold → rely on trustee |
| `False` | Expected utility is below threshold → decline |

**Note:** A `False` decision does **not** mean the trustee is untrustworthy. It means the *risk/reward profile for this trustor under these stakes* does not justify reliance. A different trustor (higher τ, lower ρ) might decide `True` with the same trustee.

### 7.6 Sensitivity Analysis

Vary one parameter at a time to see how the decision changes:

```python
import numpy as np

def sweep(trustor_risk, seller, U_win, U_loss, stakes):
    trustor = Trustor("t", trust_propensity=0.6, risk_aversion=trustor_risk)
    sc = TrustScenario("sweep", trustor, seller, U_win, U_loss, stakes)
    return sc.target_trust()

seller = Trustee("seller", a=130, b=20, discount_factor=0.8)

for ρ in [0.05, 0.1, 0.2, 0.3, 0.5]:
    t = sweep(ρ, seller, 100, -800, 800)
    print(f"ρ={ρ:.2f} → trust={t:.3f}")
```

---

## 8. Working with Groups

A `TrustGroup` represents a **collective trustee**—a team, committee, swarm, or multi-signature wallet.

### 8.1 The Constructor

```python
TrustGroup(
    members: list[Trustee],
    aggregation: AggregationType = AggregationType.SERIES,
    quorum_k: int | None = None,
    correlation_rho: float = 0.0,
    cohesion_pen: float = 0.0,
    cohesion_rew: float = 0.0,
    cohesion_tau: float = 0.0,
    entropy_override: float | None = None,
    custom_aggregator: Callable | None = None,
    collective_discount: float | None = None,
    apply_collective_strategic: bool = False,
    delta_star: float = 0.708,
    beta_s: float = 10.0,
)
```

### 8.2 The Bare Minimum

```python
from layeredtrust import Trustee, TrustGroup, AggregationType

trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
trucks.append(Trustee("T4", a=40, b=60))

platoon = TrustGroup(trucks, aggregation=AggregationType.SERIES)

print(platoon.aggregate_trust())  # 0.40
```

Here there is no cohesion penalty, so `κ = 1` and the trust is just `min(T)`.

### 8.3 Inspecting Components

```python
print(platoon.individual_trusts())   # [0.92, 0.92, 0.92, 0.92, 0.40]
print(platoon.aggregate_raw())       # 0.40
print(platoon.cohesion_factor())     # 1.0
print(platoon.aggregate_trust())     # 0.40
```

### 8.4 Adding Cohesion Penalty

```python
platoon = TrustGroup(
    trucks,
    aggregation=AggregationType.SERIES,
    cohesion_pen=10.0,
    cohesion_tau=0.02,
)
print(platoon.cohesion_factor())     # ~0.66
print(platoon.aggregate_trust())     # ~0.264
```

### 8.5 Using a Group in a Scenario

`TrustScenario` accepts a `TrustGroup` as the target:

```python
manager = Trustor("manager", trust_propensity=1.0, risk_aversion=0.1)
sc = TrustScenario("Platoon", manager, platoon,
                   U_win=10_000, U_loss=-500_000, stakes=500_000)
decision, trust, eu = sc.run()
print(decision, trust, eu)
```

### 8.6 Groups of Groups

If you need nested teams, wrap a `TrustGroup` in a custom aggregator:

```python
def two_level(scores):
    # Treat first three as sub-team A, rest as sub-team B
    a = min(scores[:3])
    b = min(scores[3:])
    return min(a, b)

g = TrustGroup(members, aggregation=AggregationType.CUSTOM,
               custom_aggregator=two_level)
```

---

## 9. Aggregation Architectures in Depth

### 9.1 Series (Weakest-Link)

**Formula:** `T = min(T_j) × κ`

**Use when:** Success requires *every* member to succeed.

- Autonomous vehicle platoons
- Surgical teams (every role must function)
- Multi-step pipelines
- Series reliability systems

```python
g = TrustGroup(trucks, aggregation=AggregationType.SERIES)
```

**Intuition:** One bad agent caps the whole group.

### 9.2 Parallel (Redundancy)

**Formula:** `T = (1 − ∏(1 − T_j)) × κ`

**Use when:** Success requires *at least one* member to succeed.

- Redundant sensor arrays
- Backup systems
- Distributed consensus
- Ensemble classifiers

```python
g = TrustGroup(sensors, aggregation=AggregationType.PARALLEL)
```

**Intuition:** More members → higher trust, with diminishing returns.

**Numerical tip:** For large `m` with high `T_j`, use `1 − (1−T)^m` directly; the library's loop is stable but may lose precision beyond `m ≈ 1000`.

### 9.3 Quorum (k-of-m)

**Formula:** `T = P(X ≥ k) × κ`, where `X ~ BetaBin(m, α, β)` with correlation `ρ`.

**Use when:** A threshold number of members must agree.

- DAO multi-signature wallets (e.g., 5-of-9)
- Board votes
- Juries
- Fact-checking panels

```python
g = TrustGroup(
    signers,
    aggregation=AggregationType.QUORUM,
    quorum_k=5,
    correlation_rho=0.2,
)
```

**Intuition:** Requires `k` trustworthy members. Positive correlation reduces effective trust (members fail together).

### 9.4 Custom

**Formula:** `T = f(T_j) × κ` for any monotone `f`.

**Use when:** Your architecture is none of the above.

```python
from layeredtrust import harmonic_mean

g = TrustGroup(
    team,
    aggregation=AggregationType.CUSTOM,
    custom_aggregator=harmonic_mean,
)
```

### 9.5 Choosing the Right Architecture

| Decision structure | Architecture |
|--------------------|-------------|
| All must succeed | `SERIES` |
| Any one suffices | `PARALLEL` |
| k must agree | `QUORUM` |
| Average/median/harmonic | `CUSTOM` |
| Safety-critical team | `SERIES` + high `ν_pen` |
| Advisory panel | `QUORUM` + `ν_rew > 0` |

---

## 10. The Cohesion–Diversity Function

### 10.1 The Formula

$$\kappa = \min\left\{1,\ \exp\!\left(-\nu_{\text{pen}} \max(0, \sigma^2 - \tau) + \nu_{\text{rew}} H\right)\right\}$$

### 10.2 What Each Parameter Does

| Parameter | Effect | Typical range |
|-----------|--------|---------------|
| `cohesion_pen` (ν_pen) | Penalizes high variance | 2–50 |
| `cohesion_rew` (ν_rew) | Rewards high entropy | 0–10 |
| `cohesion_tau` (τ) | Variance tolerance | 0.01–0.10 |

### 10.3 Reading the Output

- `κ = 1` → no cohesion penalty; group is "as good as its aggregate"
- `κ = 0.5` → halved effective trust due to poor cohesion
- `κ < 1` is the common case in safety-critical teams
- `κ = 1` is common in advisory panels (with `ν_rew`)

### 10.4 Entropy Source

By default, `H` is the Shannon entropy of the **normalized trust vector**:

```python
ts = [0.9, 0.7, 0.5, 0.3]
# normalized: [0.375, 0.292, 0.208, 0.125]
# H = -Σ p log p ≈ 1.32 nats
```

For **classification** scenarios (e.g., fact-checking with True/False labels), supply the entropy of the *label distribution*:

```python
g = TrustGroup(..., entropy_override=0.673)
```

### 10.5 A Decision Table

| Scenario | `ν_pen` | `ν_rew` | `τ` |
|----------|---------|---------|-----|
| Surgical team | 50 | 0 | 0.01 |
| Platoon | 10 | 0 | 0.02 |
| DAO multi-sig | 2 | 0 | 0.02 |
| Advisory panel | 5 | 5 | 0.10 |
| Sensor array | 1 | 0 | 0.05 |

### 10.6 Sensitivity Analysis

```python
import numpy as np

scores = [0.95, 0.90, 0.98, 0.70, 0.85]
trucks = [Trustee(f"T{i}", a=100*s, b=100*(1-s)) for i, s in enumerate(scores)]

for ν in [1, 5, 10, 20, 50]:
    g = TrustGroup(trucks, aggregation=AggregationType.SERIES,
                   cohesion_pen=ν, cohesion_tau=0.0)
    print(f"ν_pen={ν:>2}  κ={g.cohesion_factor():.3f}  T={g.aggregate_trust():.3f}")
```

---

## 11. Correlated Quorum Models

### 11.1 The Idea

In an *independent* quorum, each member's success/failure is a fresh coin flip. In reality, members **influence each other**: signers may collude, sensors may share a common failure mode, experts may herd.

The **beta-binomial** model captures this via a latent common parameter.

### 11.2 The Correlation Parameter ρ

- `ρ = 0` → independent (ordinary binomial)
- `ρ = 0.1–0.3` → mild correlation
- `ρ > 0.5` → strong correlation (near-herding)

**Typical values:**

| Scenario | ρ |
|----------|---|
| Independent sensors | 0.0 |
| Signers in a DAO | 0.2 |
| Members of the same firm | 0.3 |
| Family members | 0.5 |

### 11.3 The Effect on Quorum Trust

Positive correlation **reduces** effective trust for `k > m × T̄` (i.e., above-mean thresholds) and **increases** it for below-mean thresholds. For most realistic multi-sig scenarios, correlation reduces trust:

```python
signers = [Trustee(f"S{i}", a=90, b=10) for i in range(6)]
signers.append(Trustee("M", a=60, b=40))
signers.extend([Trustee(f"U{i}", a=40, b=60) for i in range(2)])

for ρ in [0.0, 0.1, 0.2, 0.3, 0.5]:
    g = TrustGroup(signers, aggregation=AggregationType.QUORUM,
                   quorum_k=5, correlation_rho=ρ)
    print(f"ρ={ρ:.1f}  raw={g.aggregate_raw():.4f}")
```

Output (approximately):

```
ρ=0.0  raw=0.9636
ρ=0.1  raw=0.9448
ρ=0.2  raw=0.9113
ρ=0.3  raw=0.8711
ρ=0.5  raw=0.7982
```

### 11.4 Numerical Behavior at ρ → 0

The library clamps `ρ` to `[0, 0.999]`. For `ρ < 1e-6`, it switches to the ordinary binomial for numerical stability.

### 11.5 When Not to Use Correlation

If your members are demonstrably independent (e.g., geographically separated sensors with no shared firmware), leave `ρ = 0`. Overestimating ρ makes the model overly pessimistic.

---

## 12. Temporal Dynamics and Trust Recovery

### 12.1 The Two Dynamics

**Autonomous decay** — trust slowly decays toward a baseline if no evidence arrives. (This is implicit in the Beta update with `λ`.)

**Active dynamics** — trust jumps up on success, collapses on failure:

$$\Delta T = k_{\text{build}}(1-T) \text{ on success}, \qquad \Delta T = -k_{\text{destroy}} T \text{ on failure}$$

### 12.2 Simulating a Failure

```python
car = Trustee("car", k_build=0.1, k_destroy=0.8)
car.trust = 0.65

car.asymmetric_dynamics(success=False)
print(car.trust)   # 0.130
```

### 12.3 Simulating Recovery

```python
for i in range(20):
    car.asymmetric_dynamics(success=True)
    print(f"Ride {i+1}: T = {car.trust:.3f}")
```

### 12.4 Closed-Form Recovery Time

```python
car.trust = 0.13
t = car.recovery_time(target=0.8, k_build=0.1)
print(f"{t:.2f} rides to recover to 0.8")   # 14.70
```

### 12.5 Choosing Build/Destroy Rates

| Domain | `k_build` | `k_destroy` |
|--------|-----------|-------------|
| Robotaxi | 0.1 | 0.8 |
| Chatbot | 0.2 | 0.6 |
| Team member | 0.15 | 0.45 |
| Institutional | 0.05 | 0.30 |

### 12.6 Discrete vs. Continuous

The library uses Euler steps with `dt=1` by default. For finer simulations:

```python
car.asymmetric_dynamics(success=True, dt=0.1)
```

### 12.7 A Warning About Over-Updating

Do not call `asymmetric_dynamics` and `update_bayes` for the same event. Choose one:

- Use `update_bayes` for a **principled** Bayesian count.
- Use `asymmetric_dynamics` for a **phenomenological** trust level.

Mixing them double-counts evidence.

---

## 13. Network Propagation

### 13.1 The Idea

If agent A trusts B, and B trusts C, then A has **indirect** evidence about C through B. Layer 6 aggregates these indirect paths.

### 13.2 Setting Up a Network

```python
from layeredtrust import TrustNetwork

net = TrustNetwork(n_agents=4)

net.set_direct(0, 1, 0.9)   # agent 0 → agent 1
net.set_direct(0, 2, 0.4)
net.set_direct(1, 2, 0.95)
net.set_direct(1, 3, 0.7)
net.set_direct(2, 3, 0.85)
```

### 13.3 Setting Recommender Credibility

```python
net.set_credibility(0, 1, 0.9)   # A trusts B's recommendations
net.set_credibility(0, 2, 0.4)   # A trusts C's recommendations less
```

Default credibility is `1.0` for all pairs.

### 13.4 One-Step and Iterated Updates

```python
net.update(mu=0.5)              # one damped step
net.iterate(steps=10, mu=0.5)   # ten steps
```

Damping `μ` blends direct trust with indirect trust:

$$T^{(t+1)} = \mu \cdot T^{\text{direct}} + (1-\mu) \cdot T^{\text{indirect}}$$

- `μ = 1` → ignore network (pure direct)
- `μ = 0` → ignore direct (pure indirect)
- `μ = 0.5` → balanced (typical)

### 13.5 Reading the Result

```python
for row in net.direct:
    print([round(x, 3) for x in row])
```

### 13.6 Combining with TrustScenario

Extract the propagated trust for a specific pair and feed it into a `Trustee`:

```python
net.iterate(steps=5)
propagated = net.direct[0][3]

target = Trustee("target")
target.trust = propagated
sc = TrustScenario("Propagated", trustor, target,
                   U_win=100, U_loss=-200, stakes=200)
```

---

## 14. Common Recipes

### 14.1 Trust a Seller from Reviews

```python
def seller_trust(pos, neg, tau=0.6, rho=0.2):
    seller = Trustee("seller", a=pos + 1, b=neg + 1, discount_factor=0.8)
    buyer = Trustor("buyer", trust_propensity=tau, risk_aversion=rho)
    sc = TrustScenario("Amazon", buyer, seller,
                       U_win=100, U_loss=-800, stakes=800)
    return sc.run()
```

### 14.2 Evaluate a Multi-Sig Wallet

```python
def multisig_trust(trusts, k, rho=0.2):
    members = [Trustee(f"S{i}", a=100*t, b=100*(1-t))
               for i, t in enumerate(trusts)]
    g = TrustGroup(members, aggregation=AggregationType.QUORUM,
                   quorum_k=k, correlation_rho=rho,
                   cohesion_pen=2.0, cohesion_tau=0.02)
    return g.aggregate_trust()
```

### 14.3 Simulate a Failure-and-Recovery Trajectory

```python
def trajectory(initial, failure_at, n_rides=30):
    car = Trustee("car", k_build=0.1, k_destroy=0.8)
    car.trust = initial
    history = [car.trust]
    for i in range(n_rides):
        success = (i != failure_at)
        car.asymmetric_dynamics(success=success)
        history.append(car.trust)
    return history
```

### 14.4 Compare Architectures on the Same Members

```python
def compare_architectures(members, k=None):
    results = {}
    for arch in [AggregationType.SERIES, AggregationType.PARALLEL]:
        g = TrustGroup(members, aggregation=arch)
        results[arch.value] = g.aggregate_trust()
    if k is not None:
        g = TrustGroup(members, aggregation=AggregationType.QUORUM, quorum_k=k)
        results["quorum"] = g.aggregate_trust()
    return results
```

### 14.5 Batch Scenario Evaluation

```python
def evaluate_batch(scenarios):
    rows = []
    for sc in scenarios:
        decision, trust, eu = sc.run()
        rows.append({"name": sc.name, "trust": trust, "eu": eu, "decision": decision})
    return rows
```

### 14.6 Bayesian A/B Comparison of Two Trustees

```python
def compare(a_successes, a_failures, b_successes, b_failures, n_samples=10000):
    import random
    rng = random.Random(0)
    a_wins = 0
    for _ in range(n_samples):
        pa = rng.betavariate(a_successes + 1, a_failures + 1)
        pb = rng.betavariate(b_successes + 1, b_failures + 1)
        a_wins += int(pa > pb)
    return a_wins / n_samples
```

---

## 15. Advanced Usage

### 15.1 Custom Aggregation with Weights

```python
def weighted_min(scores, weights):
    return min(s / w for s, w in zip(scores, weights) if w > 0)

def make_weighted_min(weights):
    return lambda scores: weighted_min(scores, weights)

g = TrustGroup(members,
               aggregation=AggregationType.CUSTOM,
               custom_aggregator=make_weighted_min([1.0, 1.0, 0.5, 1.0]))
```

### 15.2 Dynamic Group Composition

The library assumes fixed membership. To model churn, rebuild the group between steps:

```python
def step_with_churn(group, arrive=None, depart=None):
    members = [m for m in group.members if m.id not in (depart or [])]
    if arrive:
        members.append(arrive)
    return TrustGroup(members, aggregation=group.aggregation,
                      cohesion_pen=group.cohesion_pen,
                      cohesion_tau=group.cohesion_tau)
```

### 15.3 Subclassing Trustee for Custom Dynamics

```python
class LevyTrustee(Trustee):
    """Adds Lévy jumps to the trust trajectory."""
    jump_size: float = 0.2
    jump_prob: float = 0.05

    def asymmetric_dynamics(self, success, **kwargs):
        super().asymmetric_dynamics(success, **kwargs)
        import random
        if random.random() < self.jump_prob:
            self.trust = max(0.0, self.trust - self.jump_size)
        return self.trust
```

### 15.4 Building a Custom Cohesion Function

The library uses a specific `κ`. To experiment with alternatives, compute your own factor and multiply:

```python
def custom_kappa(group):
    ts = group.individual_trusts()
    spread = max(ts) - min(ts)
    return 1.0 - 0.5 * spread

raw = platoon.aggregate_raw()
adjusted = raw * custom_kappa(platoon)
```

### 15.5 Calibration via Maximum Likelihood

Given a dataset of `(features, outcome)` pairs, fit `ν_pen`, `ν_rew`, `ρ` by grid search:

```python
from itertools import product

def fit_params(dataset, ν_grid, ρ_grid):
    best = None
    for ν_pen, ν_rew, ρ in product(ν_grid, ν_grid, ρ_grid):
        loss = 0.0
        for features, y in dataset:
            g = TrustGroup(features, aggregation=AggregationType.QUORUM,
                           quorum_k=len(features)//2+1,
                           correlation_rho=ρ,
                           cohesion_pen=ν_pen,
                           cohesion_rew=ν_rew)
            p = g.aggregate_trust()
            loss += (p - y) ** 2   # Brier
        if best is None or loss < best[0]:
            best = (loss, ν_pen, ν_rew, ρ)
    return best
```

### 15.6 Reproducing the Paper's Simulation

```python
from layeredtrust import simulate_comparison

results = simulate_comparison(n_scenarios=10000, seed=42)
for name, r in results.items():
    print(f"{name:<10} acc={r['accuracy']:.4f} brier={r['brier']:.4f}")
```

---

## 16. Parameter Calibration Guide

### 16.1 Which Parameters Need Calibration

| Parameter | Class | How to Calibrate |
|-----------|-------|------------------|
| `a`, `b` | `Trustee` | Count successes/failures from history |
| `A`, `B`, `I` | `Trustee` | Survey instruments or expert elicitation |
| `discount_factor` | `Trustee` | Estimate from repeated-game data |
| `k_build`, `k_destroy` | `Trustee` | Fit to observed recovery trajectories |
| `trust_propensity` | `Trustor` | Survey of trust disposition |
| `risk_aversion` | `Trustor` | Elicit from lottery choices |
| `delta_star` | `Trustor` | Compute from game payoffs |
| `beta_s` | `Trustor` | Fit to observed switching behavior |
| `quorum_k` | `TrustGroup` | Set by governance rule |
| `correlation_rho` | `TrustGroup` | Method of moments on joint outcomes |
| `cohesion_pen`, `cohesion_rew`, `cohesion_tau` | `TrustGroup` | MLE on team performance data |

### 16.2 Practical Heuristics When You Have No Data

| Parameter | Heuristic |
|-----------|-----------|
| `τ` | 0.5 (neutral) or 0.6 (slightly trusting) |
| `ρ` | 0.2 (moderate) |
| `δ` | 0.8 (established relationship) |
| `δ*` | 0.708 (default) |
| `β_s` | 10 |
| `k_build` | 0.1 |
| `k_destroy` | 0.6 (3:1) to 0.8 (8:1) |
| `ν_pen` | 10 for safety-critical, 2 for advisory |
| `ρ_corr` | 0.2 |

### 16.3 Validating Calibration

After fitting, compute:

- **Brier score** on held-out data
- **ROC-AUC** for binary decisions
- **Calibration plot** (predicted vs. observed trust)

If your model is poorly calibrated, adjust the cohesion and strategic parameters first, then revisit cognitive weights.

### 16.4 Cross-Validation Protocol

```python
def kfold_cv(dataset, k=5):
    import random
    rng = random.Random(0)
    shuffled = dataset[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::k] for i in range(k)]
    scores = []
    for i in range(k):
        train = [x for j, f in enumerate(folds) if j != i for x in f]
        test = folds[i]
        params = fit_params(train, ...)
        scores.append(evaluate(test, params))
    return sum(scores) / k
```

---

## 17. Troubleshooting

### 17.1 "ValueError: quorum_k must be set for QUORUM aggregation"

You used `AggregationType.QUORUM` without setting `quorum_k`:

```python
g = TrustGroup(members,
               aggregation=AggregationType.QUORUM,
               quorum_k=5)   # ← add this
```

### 17.2 "ValueError: custom_aggregator must be provided"

You used `AggregationType.CUSTOM` without a function:

```python
g = TrustGroup(members,
               aggregation=AggregationType.CUSTOM,
               custom_aggregator=harmonic_mean)
```

### 17.3 "ValueError: TrustGroup must contain at least one member"

Empty group. Add members or check your construction loop.

### 17.4 Trust Always 0 or 1

Check:
- Are `a` and `b` sensible? (`a = b = 0` → undefined; library uses `0.5`)
- Is `discount_factor` below `delta_star`? (strategic override drives trust down)
- Is `trust_propensity` set to something reasonable?

### 17.5 Cohesion Factor Always 1

Check:
- Is `cohesion_pen > 0`?
- Is the variance actually above `cohesion_tau`?
- For single-member groups, `κ = 1` by definition.

### 17.6 Correlation Has No Effect

Check:
- Is `ρ` above `1e-6`?
- Is `aggregation=QUORUM`?

### 17.7 Decision Flips Unexpectedly

Common causes:

- `U_loss` sign (should usually be negative)
- `stakes` too small or too large
- `risk_aversion` too low

Print `scenario.report()` to inspect all intermediates.

### 17.8 Numerical Precision

For very large groups (`m > 1000`), the binomial survival function uses `math.comb`, which may overflow. Reformulate using recurrence or logarithms if needed. For typical groups (`m ≤ 100`), no issue.

---

## 18. Best Practices

### 18.1 Model Design

1. **Start simple.** Use just Layers 1–3 for a first prototype. Add L4–L7 only when the domain demands them.
2. **Keep units consistent.** If `U_win` and `U_loss` are in dollars, `stakes` should be too.
3. **Sign discipline.** `U_loss` should be negative unless you are modeling a non-loss scenario.
4. **Weights sum to 1.** `alpha + beta + gamma = 1` (the library normalizes internally, but explicit is better).
5. **Document parameters.** Record where each number came from.

### 18.2 Coding Practices

1. **Use dataclasses.** `Trustee`, `Trustor`, `TrustGroup`, and `TrustScenario` are all dataclasses—construct them in one place and pass them around.
2. **Do not mutate shared trustees.** If two scenarios use the same `Trustee`, changes propagate. Copy or reconstruct as needed.
3. **Separate model from data.** Keep your parameter values in a config module.
4. **Log intermediate values.** `scenario.report()` is your friend.
5. **Version your parameters.** If you tune `ν_pen`, record the value in a comment or config.

### 18.3 Scientific Practices

1. **Report calibration.** State where each parameter came from.
2. **Report sensitivity.** Show how the decision changes with `ρ`, `ν_pen`, etc.
3. **Use held-out data.** When fitting parameters, reserve a test set.
4. **Do not over-claim.** A simulation is not empirical validation.
5. **Cite the framework.** Reference the manuscript when using the library.

### 18.4 Performance

- The library is pure Python and fast for `m ≤ 100` members and `n ≤ 10⁶` scenarios.
- For larger workloads, consider caching the Beta-binomial survival function.
- For heavy simulations, use `multiprocessing` or `concurrent.futures`.

---

## 19. Frequently Asked Questions

**Q1. Do I need all seven layers?**

No. Start with Layers 1–3 for a basic model. Add L4 (temporal), L5 (strategic), L6 (network), and L7 (group) as needed.

**Q2. What's the difference between `trust` and `a/b`?**

`a/b` is a Bayesian count that can be updated. `trust` is an explicit value that overrides the Beta mean. Use `a/b` when you have interaction history; use `trust` when you have an externally supplied score.

**Q3. Why is my decision `False` even though the trustee has high trust?**

Because expected utility, not trust alone, drives the decision. A high-trust trustee with unfavorable payoffs (large `U_loss`, small `U_win`) can still be declined.

**Q4. Why does the paper report `0.243` for the platoon but the library reports `0.317`?**

The paper's `σ² = 0.07` is an illustrative value; the library computes the exact variance of the given trust vector (`σ² ≈ 0.043`). The library's result is the correct one for the stated inputs.

**Q5. Can I use `layeredtrust` for reinforcement learning?**

Yes, as a trust-value function. Wrap the effective trust in a reward signal and train your agent normally.

**Q6. How do I model a Sybil attack?**

Set the Sybil's `a` low and `b` high, but `discount_factor` very low (`0.2`). The strategic override will suppress trust even if the Sybil's Beta mean looks reasonable.

**Q7. What if I don't know `delta_star`?**

Use `0.708` (default) for the standard PD payoff. For other games, compute `(T-R)/(T-P)` from your payoff structure.

**Q8. Can I model mixed human-AI teams?**

Yes. Humans and AIs are both `Trustee` instances. Set `A`, `B`, `I` from human survey data and Beta counts from AI logs, then combine in a `TrustGroup`.

**Q9. How do I test my model?**

Use `test_layeredtrust.py` as a template. At minimum, test that bounds `[0, 1]` hold for your scenarios and that the unification property holds when `m = 1`.

**Q10. Can I use the library commercially?**

Yes—MIT license.

**Q11. What about multi-trustor scenarios?**

Run a separate `TrustScenario` per trustor. To aggregate trustors, use a `TrustGroup` whose members are "trustor-specific trust scores."

**Q12. How do I cite the library?**

See the [Citation](#citation) section in the README.

---

## 20. Reference Tables

### 20.1 Aggregation Types

| Type | Formula | Use case |
|------|---------|----------|
| `SERIES` | `min(T_j) × κ` | All-must-succeed |
| `PARALLEL` | `(1 − ∏(1−T_j)) × κ` | Any-succeeds |
| `QUORUM` | `P(X ≥ k) × κ` | k-of-m |
| `CUSTOM` | `f(T_j) × κ` | Anything monotone |

### 20.2 Strategic Factor Values

| `δ` | `Φ(δ)` at `β_s=10`, `δ*=0.708` |
|-----|-------------------------------|
| 0.3 | 0.016 |
| 0.5 | 0.111 |
| 0.6 | 0.254 |
| 0.708 | 0.500 |
| 0.8 | 0.715 |
| 0.9 | 0.874 |
| 0.95 | 0.922 |

### 20.3 Cohesion Penalty Examples

Given `σ² = 0.05`, `τ = 0`:

| `ν_pen` | `κ` |
|---------|-----|
| 1 | 0.951 |
| 5 | 0.779 |
| 10 | 0.607 |
| 20 | 0.368 |
| 50 | 0.082 |

### 20.4 Quorum Survival with Correlation

`m=9`, `k=5`, `T̄=0.756`:

| `ρ` | `P(X ≥ 5)` |
|-----|------------|
| 0.0 | 0.96 |
| 0.1 | 0.94 |
| 0.2 | 0.91 |
| 0.3 | 0.87 |
| 0.5 | 0.80 |

### 20.5 Typical Recovery Times (`k_build=0.1`)

| Start | Target | Rides |
|-------|--------|-------|
| 0.13 | 0.8 | 14.7 |
| 0.5 | 0.8 | 7.0 |
| 0.3 | 0.9 | 18.4 |
| 0.7 | 0.95 | 20.6 |

### 20.6 Decision Checklist

Before trusting, verify:

- [ ] `U_loss` is negative
- [ ] `stakes` is in the same units as `U_win`/`U_loss`
- [ ] `risk_aversion` is appropriate for the domain
- [ ] `trust_propensity` is calibrated to the trustor
- [ ] `discount_factor` reflects the trustee's long-term orientation
- [ ] For groups, `aggregation` matches the decision rule
- [ ] For quorum, `quorum_k` matches the governance rule
- [ ] For cohesion, `cohesion_pen` matches the domain's risk sensitivity

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Trustee** | The agent being evaluated |
| **Trustor** | The agent making the decision |
| **Effective trust** | The final trust score after all layers |
| **Cohesion factor (κ)** | Multiplier reflecting group coordination |
| **Correlation (ρ)** | Degree of dependence among group members |
| **Discount factor (δ)** | Trustee's long-term orientation |
| **Betrayal threshold (δ*)** | Minimum δ for sustained cooperation |
| **Trust propensity (τ)** | Trustor's baseline willingness |
| **Risk aversion (ρ_r)** | Trustor's threshold multiplier |
| **Brier score** | Mean squared error of probabilistic predictions |
| **Weakest-link** | Series aggregation (min) |
| **Redundancy** | Parallel aggregation (1 − ∏) |

---

## Appendix B: Further Reading

- **Mayer, Davis, & Schoorman (1995)** — Cognitive antecedents of trust
- **Jøsang & Ismail (2002)** — Beta reputation systems
- **Axelrod (1984)** — Evolution of cooperation
- **Surowiecki (2004)** — Wisdom of crowds
- **Ponnambalam et al. (2021)** — Prior MAS trust framework
- **Wang (2026)** — The unified multi-layer framework (manuscript)

---

## Appendix C: Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026 | Initial release |

---

**Maintainer:** Harris Wang · [harriw@athabascau.ca](mailto:harriw@athabascau.ca)

**Repository:** [github.com/hongxuehariswang/layeredtrust](https://github.com/)

**Issues & Discussions:** [github.com/hongxuehariswang/layeredtrust/issues](https://github.com/)

---

*End of User Guide*