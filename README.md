# trustlib

> **A Unified Multi-Layer Mathematical Framework for Trust-Based Decision Making in Multi-Agent Systems (MAS)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-none-success.svg)](#installation)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#changelog)

`trustlib` implements a mathematically consistent, computationally tractable framework for trust-based decision making across both **dyadic** (one trustor, one trustee) and **collective** (one trustor, a structured group of trustees) relationships in multi-agent systems. It synthesizes cognitive psychology, Bayesian inference, game theory, dynamical systems, and network science into a single, reduction-consistent model.

This library accompanies the manuscript:

> **Wang, H.** *Trust-Based Decision-Making for Multi-Agent Systems: A Unified Multi-Layer Mathematical Framework.* Athabasca University.

---

## Table of Contents

- [Why trustlib?](#why-trustlib)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [The Seven Layers](#the-seven-layers)
  - [The Group Trust Extension (GTE)](#the-group-trust-extension-gte)
  - [The Unification Theorem](#the-unification-theorem)
- [Usage Examples](#usage-examples)
  - [1. Dyadic Trust: E-Commerce Seller](#1-dyadic-trust-e-commerce-seller)
  - [2. Dyadic Trust with Asymmetric Dynamics](#2-dyadic-trust-with-asymmetric-dynamics)
  - [3. Group Trust: Series (Weakest-Link)](#3-group-trust-series-weakest-link)
  - [4. Group Trust: Parallel (Redundancy)](#4-group-trust-parallel-redundancy)
  - [5. Group Trust: Quorum with Correlation](#5-group-trust-quorum-with-correlation)
  - [6. Group Trust: Custom Aggregation](#6-group-trust-custom-aggregation)
  - [7. Network Propagation](#7-network-propagation)
  - [8. Proof-of-Concept Simulation](#8-proof-of-concept-simulation)
- [API Reference](#api-reference)
  - [Trustee](#trustee)
  - [Trustor](#trustor)
  - [TrustGroup](#trustgroup)
  - [TrustScenario](#trustscenario)
  - [TrustNetwork](#trustnetwork)
  - [AggregationType](#aggregationtype)
  - [Helper Functions](#helper-functions)
- [Mathematical Background](#mathematical-background)
- [Reproducing the Paper's Case Studies](#reproducing-the-papers-case-studies)
- [Testing](#testing)
- [Design Decisions and Caveats](#design-decisions-and-caveats)
- [Extending the Library](#extending-the-library)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Why trustlib?

Most computational trust models focus on **dyadic** relationships: one trustor assessing a single trustee. Real-world multi-agent systems, however, routinely demand trust in **groups**—surgical teams, autonomous vehicle platoons, DAO multi-signature wallets, hybrid human-AI fact-checking collectives—where success depends on internal coordination, heterogeneity, and decision rules (quorums, weakest-link dependencies, redundancy).

`trustlib` bridges this gap:

| Capability | Dyadic models | `trustlib` |
|-----------|:-------------:|:----------:|
| Individual trustee | ✅ | ✅ |
| **Structured group trustee** | ❌ | ✅ |
| Cognitive antecedents (A, B, I) | partial | ✅ |
| Bayesian learning with recency | ✅ | ✅ |
| Asymmetric temporal dynamics | ❌ | ✅ |
| Strategic (game-theoretic) override | ❌ | ✅ |
| Network propagation | partial | ✅ |
| Correlated failures in quorum | ❌ | ✅ |
| Cohesion/diversity adjustment | ❌ | ✅ |
| Formal reduction to dyadic | — | ✅ |
| **Zero dependencies** | — | ✅ |

---

## Features

- **Seven composable layers** — from cognitive antecedents to coupled dynamics.
- **Group Trust Extension** with four aggregation architectures: `SERIES`, `PARALLEL`, `QUORUM`, `CUSTOM`.
- **Correlated quorum model** via the beta-binomial distribution (intraclass correlation `rho`).
- **Cohesion–diversity function** with penalty and reward terms, plus a tunable tolerance threshold.
- **Smooth logistic strategic override** — no discontinuous thresholds.
- **Network propagation** with damping and recommender credibility.
- **End-to-end `TrustScenario`** pipeline: cognitive → strategic → utility → decision.
- **Proof-of-concept simulation study** comparing the framework against baseline heuristics.
- **Zero external dependencies** — pure Python standard library only.
- **Fully tested** — 40+ unit tests covering limits, bounds, and edge cases.
- **Extensible** — plug in your own aggregators, entropy measures, and dynamics.

---

## Installation

`trustlib` has **no external dependencies** and requires Python 3.9+.

### Option 1 — Copy the single file

Drop `trustlib.py` into your project and import it:

```python
from trustlib import Trustee, Trustor, TrustGroup, TrustScenario
```

### Option 2 — Install as a package

```bash
git clone https://github.com/<your-username>/trustlib.git
cd trustlib
pip install -e .
```

### Option 3 — Development install

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from trustlib import Trustee, Trustor, TrustScenario

# A buyer (trustor) evaluates an Amazon seller (trustee)
buyer  = Trustor("buyer", trust_propensity=0.6, risk_aversion=0.2)
seller = Trustee("seller", a=130, b=20, discount_factor=0.8)

scenario = TrustScenario(
    "Amazon Seller", buyer, seller,
    U_win=100, U_loss=-800, stakes=800,
)
decision, trust, eu = scenario.run()

print(f"Effective trust:  {trust:.3f}")
print(f"Expected utility: ${eu:.2f}")
print(f"Decision:         {'BUY' if decision else 'DO NOT BUY'}")

# Effective trust:  0.372
# Expected utility: $-465.20
# Decision:         DO NOT BUY
```

For a group trustee:

```python
from trustlib import TrustGroup, AggregationType

trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
trucks.append(Trustee("T4", a=40, b=60))  # sensor glitch

platoon = TrustGroup(
    trucks,
    aggregation=AggregationType.SERIES,
    cohesion_pen=10.0,
    cohesion_tau=0.02,
)
print(f"Platoon trust: {platoon.aggregate_trust():.3f}")  # 0.317
```

---

## Core Concepts

### The Seven Layers

| Layer | Name | Role |
|-------|------|------|
| **L1** | Perceived Trustworthiness | Weighted sum of ability (`A`), benevolence (`B`), integrity (`I`). |
| **L2** | Behavioral Decision Threshold | Expected utility vs. `rho * stakes`. |
| **L3** | Bayesian Learning | Beta-Binomial update with recency weight `lambda`. |
| **L4** | Temporal Dynamics | Asymmetric build/destroy rates (`k_build`, `k_destroy`). |
| **L5** | Strategic Layer | Smooth logistic override of the repeated-game threshold. |
| **L6** | Network Propagation | Damped indirect trust through recommenders. |
| **L7** | Group Trust Extension | Series / Parallel / Quorum / Custom aggregation. |

### The Group Trust Extension (GTE)

A trustor's confidence in a group $G = \{j_1, \dots, j_m\}$ depends on the group's **decision architecture**:

**Series (weakest-link)**
$$T_{i,G}^{\text{series}} = \min_{j \in G}(T_{i,j}) \cdot \kappa$$

**Parallel (redundancy)**
$$T_{i,G}^{\text{parallel}} = \left(1 - \prod_{j \in G}(1 - T_{i,j})\right) \cdot \kappa$$

**Quorum (k-of-m, correlated)**
$$T_{i,G}^{\text{quorum}} = P(X \geq k) \cdot \kappa, \quad X \sim \text{BetaBin}(m, \alpha, \beta)$$

with $\alpha = \bar{T}(1-\rho)/\rho$ and $\beta = (1-\bar{T})(1-\rho)/\rho$.

**Cohesion–diversity factor**
$$\kappa = \min\left\{1,\ \exp\!\left(-\nu_{\text{pen}} \cdot \max(0, \sigma^2 - \tau) + \nu_{\text{rew}} \cdot H\right)\right\}$$

where $\sigma^2$ is the variance of individual trust scores and $H$ is their Shannon entropy.

### The Unification Theorem

> For $m = 1$, the Group Trust Extension reduces **exactly** to the dyadic model.

This is verified in `test_trustlib.py::TestUnification`.

---

## Usage Examples

### 1. Dyadic Trust: E-Commerce Seller

```python
from trustlib import Trustee, Trustor, TrustScenario

buyer  = Trustor("buyer", trust_propensity=0.6, risk_aversion=0.2)
seller = Trustee("seller", a=130, b=20, discount_factor=0.8)

scenario = TrustScenario("Amazon", buyer, seller,
                         U_win=100, U_loss=-800, stakes=800)
print(scenario.report())
# {'name': 'Amazon', 'effective_trust': 0.372,
#  'expected_utility': -465.2, 'threshold': 160.0, 'decision': False}
```

### 2. Dyadic Trust with Asymmetric Dynamics

```python
from trustlib import Trustee

car = Trustee("robotaxi", a=10, b=1, discount_factor=0.9)
car.trust = 0.65

# A single "phantom brake" failure
car.asymmetric_dynamics(success=False, k_build=0.1, k_destroy=0.8)
print(f"Trust after failure: {car.trust:.3f}")   # 0.130

# Time to recover to 0.8 under repeated success
print(f"Recovery time: {car.recovery_time(target=0.8):.2f} rides")  # 14.70
```

### 3. Group Trust: Series (Weakest-Link)

```python
from trustlib import Trustee, TrustGroup, AggregationType

trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
trucks.append(Trustee("T4", a=40, b=60))

platoon = TrustGroup(trucks,
                     aggregation=AggregationType.SERIES,
                     cohesion_pen=10.0, cohesion_tau=0.02)
print(f"Raw aggregate:    {platoon.aggregate_raw():.3f}")
print(f"Cohesion factor:  {platoon.cohesion_factor():.3f}")
print(f"Effective trust:  {platoon.aggregate_trust():.3f}")
```

### 4. Group Trust: Parallel (Redundancy)

```python
sensors = [Trustee(f"S{i}", a=70, b=30) for i in range(4)]
array = TrustGroup(sensors, aggregation=AggregationType.PARALLEL)
print(array.aggregate_trust())   # ~0.99
```

### 5. Group Trust: Quorum with Correlation

```python
signers = [Trustee(f"S{i}", a=90, b=10) for i in range(6)]
signers.append(Trustee("M", a=60, b=40))
signers.extend([Trustee(f"U{i}", a=40, b=60) for i in range(2)])

dao = TrustGroup(
    signers,
    aggregation=AggregationType.QUORUM,
    quorum_k=5,
    correlation_rho=0.2,     # signers influence each other
    cohesion_pen=2.0,
    cohesion_tau=0.02,
)
print(dao.aggregate_trust())     # ~0.87
```

### 6. Group Trust: Custom Aggregation

```python
from trustlib import TrustGroup, AggregationType, harmonic_mean

team = TrustGroup(
    [Trustee("Surgeon", a=95, b=5),
     Trustee("Anes",    a=90, b=10),
     Trustee("Nurse1",  a=98, b=2),
     Trustee("Nurse2",  a=70, b=30),
     Trustee("Perf",    a=85, b=15)],
    aggregation=AggregationType.CUSTOM,
    custom_aggregator=harmonic_mean,
    cohesion_pen=5.0,
    cohesion_tau=0.01,
)
print(team.aggregate_trust())
```

### 7. Network Propagation

```python
from trustlib import TrustNetwork

net = TrustNetwork(4)
net.set_direct(0, 1, 0.9)
net.set_direct(0, 2, 0.4)
net.set_direct(1, 2, 0.95)
net.set_direct(1, 3, 0.7)
net.set_direct(2, 3, 0.85)

net.iterate(steps=3, mu=0.5)
for row in net.direct:
    print([round(x, 3) for x in row])
```

### 8. Proof-of-Concept Simulation

```python
from trustlib import simulate_comparison

results = simulate_comparison(n_scenarios=2000, seed=42)
print(f"{'Model':<12}{'Accuracy':>12}{'Brier':>12}")
for name, r in results.items():
    print(f"{name:<12}{r['accuracy']:>12.4f}{r['brier']:>12.4f}")
```

---

## API Reference

### `Trustee`

An individual agent being evaluated.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | `str` | `"trustee"` | Identifier |
| `a`, `b` | `float` | `1.0` | Beta parameters for Bayesian reliability |
| `A`, `B`, `I` | `float?` | `None` | Perceived ability, benevolence, integrity |
| `discount_factor` | `float` | `0.9` | Repeated-game discount factor (L5) |
| `k_build` | `float` | `0.1` | Trust build rate (L4) |
| `k_destroy` | `float` | `0.8` | Trust destroy rate (L4) |
| `trust` | `float?` | `None` | Explicit override; if `None`, uses Beta mean |

| Method | Returns | Description |
|--------|---------|-------------|
| `bayes_mean()` | `float` | Posterior mean `a / (a + b)` |
| `update_bayes(success, lam=1.0)` | `None` | Recency-weighted Beta update |
| `perceived_trustworthiness(α, β, γ)` | `float` | Layer 1 weighted sum |
| `asymmetric_dynamics(success, ...)` | `float` | One Euler step of Layer 4 |
| `recovery_time(target, k_build=None)` | `float` | Closed-form recovery time |
| `current_trust()` | `float` | Explicit trust or Beta mean |
| `reset()` | `None` | Clear the explicit trust |

### `Trustor`

The agent making the decision.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `trust_propensity` | `float` | `0.5` | τ — disposition toward trust |
| `risk_aversion` | `float` | `0.2` | ρ — threshold multiplier on stakes |
| `delta_star` | `float` | `0.708` | Betrayal threshold (δ*) |
| `beta_s` | `float` | `10.0` | Logistic steepness (L5) |

### `TrustGroup`

A collective trustee.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `members` | `List[Trustee]` | required | Individual agents |
| `aggregation` | `AggregationType` | `SERIES` | Aggregation architecture |
| `quorum_k` | `int?` | `None` | Threshold for `QUORUM` |
| `correlation_rho` | `float` | `0.0` | Intraclass correlation |
| `cohesion_pen` | `float` | `0.0` | ν_pen |
| `cohesion_rew` | `float` | `0.0` | ν_rew |
| `cohesion_tau` | `float` | `0.0` | τ (variance tolerance) |
| `entropy_override` | `float?` | `None` | Override computed entropy |
| `custom_aggregator` | `Callable?` | `None` | For `CUSTOM` type |
| `collective_discount` | `float?` | `None` | Δ_G |
| `apply_collective_strategic` | `bool` | `False` | Apply Φ_G to output |

| Method | Returns | Description |
|--------|---------|-------------|
| `individual_trusts()` | `List[float]` | Current trust vector |
| `aggregate_raw()` | `float` | Aggregation without cohesion |
| `cohesion_factor()` | `float` | κ value |
| `aggregate_trust()` | `float` | `aggregate_raw() * cohesion_factor()` |
| `collective_strategic_factor(delta_star=None)` | `float` | Φ_G |
| `effective_trust()` | `float` | Complete L7 output |

### `TrustScenario`

End-to-end pipeline.

```python
TrustScenario(
    name, trustor, target,
    U_win=1.0, U_loss=-1.0, stakes=None,
    alpha=0.7, beta=0.2, gamma=0.1,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `target_trust()` | `float` | Effective trust in target |
| `expected_utility(trust)` | `float` | EU = T·U_win + (1−T)·U_loss |
| `threshold()` | `float` | θ = ρ · stakes |
| `run()` | `(bool, float, float)` | `(decision, trust, EU)` |
| `report()` | `Dict` | Full breakdown |

### `TrustNetwork`

Layer 6 propagation.

| Method | Description |
|--------|-------------|
| `set_direct(i, j, value)` | Set direct trust `i → j` |
| `set_credibility(i, k, value)` | Recommender credibility `φ_{i,k}` |
| `compute_indirect()` | Compute indirect trust |
| `update(mu=0.5)` | One damped update |
| `iterate(steps=10, mu=0.5)` | Multiple damped updates |

### `AggregationType`

```python
AggregationType.SERIES    # weakest-link
AggregationType.PARALLEL  # redundancy
AggregationType.QUORUM    # k-of-m (correlated)
AggregationType.CUSTOM    # user-supplied
```

### Helper Functions

```python
sigmoid(x)                      # Numerically stable logistic
strategic_factor(δ, δ*, β_s)    # Layer 5 override
harmonic_mean(values)
geometric_mean(values)
arithmetic_mean(values)
median(values)
simulate_comparison(n, seed, rho_true)
```

---

## Mathematical Background

### Layer 1 — Cognitive Antecedents

$$P_{i,j}(t) = \alpha A_{i,j} + \beta B_{i,j} + \gamma I_{i,j}, \quad \alpha + \beta + \gamma = 1$$

### Layer 2 — Expected Utility

$$E[U_{i,j}] = P_{i,j} \cdot U_{\text{win}} - (1 - P_{i,j}) \cdot U_{\text{loss}} \ \geq \ \theta_i = \rho_i \cdot S(t)$$

### Layer 3 — Bayesian Update (Recency-Weighted)

$$a' = \lambda a + s, \qquad b' = \lambda b + (1 - s)$$

### Layer 4 — Asymmetric Temporal Dynamics

$$\frac{dT}{dt} = k_{\text{build}}(1 - T) \cdot \mathbb{1}_{\text{success}} - k_{\text{destroy}} \cdot T \cdot \mathbb{1}_{\text{failure}}$$

### Layer 5 — Smooth Strategic Override

$$\Phi(\delta_j) = \sigma\big(\beta_s (\delta_j - \delta^*)\big), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}$$

### Layer 6 — Network Propagation

$$T^{\text{indirect}}_{i,j} = \frac{\sum_{k \neq i,j} T_{i,k} T_{k,j} \phi_{i,k}}{\sum_{k \neq i,j} T_{i,k} \phi_{i,k} + \epsilon}$$

### Layer 7 — Group Trust Extension

$$T^{\text{eff}}_{i,G} = \mathcal{F}(T_{i,G}) \cdot \kappa(\text{Var}, H) \cdot \Phi_G(\Delta_G)$$

Full derivations are in the manuscript and its appendices.

---

## Reproducing the Paper's Case Studies

The repository ships with `examples.py`, which reproduces all seven demonstrations from the manuscript:

```bash
python examples.py
```

Expected output:

```
=== Case 1: Autonomous Vehicle (Dyadic) ===
Trust after failure: 0.130
Recovery time to 0.8: 14.70 rides
{'name': 'Robo-taxi', 'effective_trust': 0.079, 'expected_utility': -180.25,
 'threshold': 15.0, 'decision': False}

=== Case 2: P2P Lending (Dyadic + Strategic) ===
...

=== Case 5: Hybrid Fact-Checking (Quorum + Diversity) ===
Cohesion factor: 1.000
Aggregate trust: 0.940
```

---

## Testing

```bash
python -m unittest test_trustlib -v
```

The test suite covers:

- **Math helpers** — sigmoid limits, binomial survival, beta-binomial → binomial limit as `rho → 0`.
- **Trustee** — Bayesian updates, recency weights, asymmetric dynamics, recovery time, bounds.
- **TrustGroup** — series, parallel, quorum (independent and correlated), custom aggregation, cohesion penalty/reward, empty-group errors.
- **Unification** — `m = 1` reduces exactly to the dyadic model.
- **TrustScenario** — decision logic, threshold computation.
- **TrustNetwork** — propagation, bounds, damping.

---

## Design Decisions and Caveats

1. **Exact variance in the cohesion factor.** The library computes the true population variance of individual trust scores, which may differ slightly from illustrative values in the manuscript.

2. **Entropy source.** By default, `H` is computed over the normalized trust vector. For classification scenarios, use `entropy_override` to pass a label-distribution entropy.

3. **Correlated quorum.** For `rho → 0`, the library falls back to the standard binomial. `rho` is clamped to `[0, 0.999]` for numerical stability.

4. **Monotonicity.** `κ` is **not** guaranteed to be monotone in each individual trust score (higher trust can increase variance). The library exposes both `aggregate_raw()` (monotone) and `aggregate_trust()` (with cohesion) so users can inspect the decomposition.

5. **Strategic factor application.** Dyadic scenarios always apply Layer 5. Group scenarios apply `Φ_G` only when `apply_collective_strategic=True`, matching the manuscript's main-text cases.

6. **Zero dependencies.** The core library uses only the Python standard library. No NumPy, no SciPy, no external math packages.

---

## Extending the Library

### Custom Aggregation Functions

```python
from trustlib import TrustGroup, AggregationType

def trimmed_mean(values, trim=0.1):
    s = sorted(values)
    k = max(1, int(len(s) * trim))
    return sum(s[k:-k]) / max(1, len(s) - 2 * k)

g = TrustGroup(
    members,
    aggregation=AggregationType.CUSTOM,
    custom_aggregator=trimmed_mean,
)
```

### Custom Dynamics

Subclass `Trustee` and override `asymmetric_dynamics`:

```python
class LevyTrustee(Trustee):
    def asymmetric_dynamics(self, success, **kwargs):
        # Custom Lévy-jump process
        ...
```

### Custom Entropy

Supply `entropy_override` to the `TrustGroup` constructor.

---

## Citation

If you use `trustlib` in your research, please cite:

```bibtex
@article{wang2026trustlib,
  author  = {Wang, Harris},
  title   = {Trust-Based Decision-Making for Multi-Agent Systems:
             A Unified Multi-Layer Mathematical Framework},
  journal = {Manuscript under review},
  year    = {2026},
  note    = {Library: \texttt{trustlib} v1.0.0}
}
```

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Add tests for any new functionality.
4. Ensure all tests pass: `python -m unittest -v`.
5. Submit a pull request.

### Code style

- PEP 8
- Type hints on all public functions
- Docstrings in Google/NumPy style
- No new external dependencies without discussion

### Reporting issues

Open an issue with:

- A minimal reproducible example
- Your Python version
- Expected vs. actual behavior

---

## Roadmap

- [ ] **v1.1** — Lévy-jump stochastic dynamics
- [ ] **v1.2** — Adversarial trust (Sybil / fake-review models)
- [ ] **v1.3** — LLM-agent trust extensions
- [ ] **v1.4** — Dynamic group composition (membership churn)
- [ ] **v1.5** — Calibration utilities (MLE for `nu_pen`, `nu_rew`, `rho`)
- [ ] **v2.0** — Vectorized backend (optional NumPy path)

---

## Acknowledgements

The framework synthesizes ideas from the Mayer–Davis–Schoorman cognitive trust model, Beta-reputation systems, repeated-game theory, dynamical systems, and network trust propagation. The author thanks the reviewers of the manuscript for their constructive feedback that shaped both the paper and this library.

---

**Maintainer:** Harris Wang · [harriw@athabascau.ca](mailto:harriw@athabascau.ca)

**Repository:** [github.com/\<your-username\>/trustlib](https://github.com/)

**Issues:** [github.com/\<your-username\>/trustlib/issues](https://github.com/)
