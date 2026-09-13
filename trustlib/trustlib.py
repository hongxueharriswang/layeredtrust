"""
trustlib — A Unified Multi-Layer Mathematical Framework for Trust-Based
Decision Making in Multi-Agent Systems (MAS).

Reference:
    Wang, H. "Trust-Based Decision-Making for Multi-Agent Systems:
    A Unified Multi-Layer Mathematical Framework."

Layers implemented:
    L1  Perceived trustworthiness          (A, B, I)
    L2  Behavioral decision threshold      (expected utility)
    L3  Bayesian learning                  (Beta-Binomial, recency)
    L4  Temporal dynamics                  (asymmetric build/destroy)
    L5  Strategic game-theoretic layer     (logistic override)
    L6  Social network propagation         (damped indirect trust)
    L7  Group Trust Extension              (series / parallel / quorum / custom)

The dyadic model is a special case of the Group Trust Extension
(Unification Theorem, Section 3.8 of the paper).

Basic usage
-----------
>>> from trustlib import Trustee, Trustor, TrustScenario
>>> buyer  = Trustor("buyer", trust_propensity=0.6, risk_aversion=0.2)
>>> seller = Trustee("seller", a=130, b=20, discount_factor=0.8)
>>> sc = TrustScenario("Amazon", buyer, seller,
...                    U_win=100, U_loss=-800, stakes=800)
>>> decision, trust, eu = sc.run()
>>> decision, round(trust, 3), round(eu, 2)
(False, 0.372, -465.2)

Group usage
-----------
>>> from trustlib import TrustGroup, AggregationType
>>> trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
>>> trucks.append(Trustee("T4", a=40, b=60))
>>> platoon = TrustGroup(trucks, aggregation=AggregationType.SERIES,
...                      cohesion_pen=10.0, cohesion_tau=0.02)
>>> round(platoon.aggregate_trust(), 3)
0.317

License: MIT
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

__all__ = [
    "AggregationType",
    "Trustee",
    "Trustor",
    "TrustGroup",
    "TrustScenario",
    "TrustNetwork",
    "harmonic_mean",
    "geometric_mean",
    "arithmetic_mean",
    "median",
    "sigmoid",
    "strategic_factor",
    "simulate_comparison",
]


# ===========================================================================
# 0. Mathematical helpers
# ===========================================================================

def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def strategic_factor(delta: float, delta_star: float = 0.708,
                     beta_s: float = 10.0) -> float:
    """
    Layer 5: smooth logistic override of the repeated-game cooperation
    threshold.

        Phi(delta) = sigma( beta_s * (delta - delta_star) )

    As beta_s -> infinity this approaches the hard indicator
    I{delta > delta_star}.
    """
    return sigmoid(beta_s * (delta - delta_star))


def _log_beta(a: float, b: float) -> float:
    """log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_binomial_pmf(k: int, m: int, alpha: float, beta: float) -> float:
    """Beta-binomial pmf: P(X = k) for X ~ BetaBin(m, alpha, beta)."""
    if k < 0 or k > m:
        return 0.0
    log_coef = (math.lgamma(m + 1)
                - math.lgamma(k + 1)
                - math.lgamma(m - k + 1))
    log_p = log_coef + _log_beta(alpha + k, beta + m - k) - _log_beta(alpha, beta)
    return math.exp(log_p)


def _binomial_sf(k: int, m: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(m, p)."""
    p = min(max(p, 0.0), 1.0)
    total = 0.0
    for q in range(k, m + 1):
        total += math.comb(m, q) * (p ** q) * ((1.0 - p) ** (m - q))
    return min(max(total, 0.0), 1.0)


def _beta_binomial_sf(k: int, m: int, alpha: float, beta: float) -> float:
    """P(X >= k) for X ~ BetaBin(m, alpha, beta)."""
    total = 0.0
    for q in range(k, m + 1):
        total += _beta_binomial_pmf(q, m, alpha, beta)
    return min(max(total, 0.0), 1.0)


def _shannon_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy (nats) of a probability vector."""
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    return h


def _variance(values: Sequence[float]) -> float:
    """Population variance."""
    n = len(values)
    if n == 0:
        return 0.0
    mu = sum(values) / n
    return sum((v - mu) ** 2 for v in values) / n


# ===========================================================================
# 1. Aggregation helpers (for CUSTOM aggregation)
# ===========================================================================

def harmonic_mean(values: Sequence[float]) -> float:
    """Harmonic mean (appropriate for weakest-link + coordination)."""
    vals = [max(v, 1e-9) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)


def geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean."""
    vals = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def arithmetic_mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    """Median (order statistic)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# ===========================================================================
# 2. Aggregation type
# ===========================================================================

class AggregationType(Enum):
    """Group decision architecture (Layer 7)."""
    SERIES   = "series"    # weakest-link, min
    PARALLEL = "parallel"  # redundancy, 1 - prod(1 - T)
    QUORUM   = "quorum"    # k-of-m, correlated via beta-binomial
    CUSTOM   = "custom"    # user-supplied monotone aggregator


# ===========================================================================
# 3. Trustee — an individual agent being evaluated
# ===========================================================================

@dataclass
class Trustee:
    """
    An individual agent (trustee) being evaluated by a trustor.

    Stores:
        - Beta parameters (a, b) for Bayesian reliability
        - Latent traits (A, B, I) for cognitive antecedents (Layer 1)
        - Discount factor for the strategic layer (Layer 5)
        - Asymmetric build/destroy rates for temporal dynamics (Layer 4)
        - An optional explicit `trust` value that overrides the Beta mean
    """
    id: str = "trustee"
    a: float = 1.0
    b: float = 1.0
    A: Optional[float] = None   # perceived ability
    B: Optional[float] = None   # perceived benevolence
    I: Optional[float] = None   # perceived integrity
    discount_factor: float = 0.9
    k_build: float = 0.1
    k_destroy: float = 0.8
    trust: Optional[float] = None

    # ----- Layer 3: Bayesian reliability --------------------------------

    def bayes_mean(self) -> float:
        """Posterior expected reliability E[T] = a / (a + b)."""
        denom = self.a + self.b
        if denom <= 0.0:
            return 0.5
        return self.a / denom

    def update_bayes(self, success: bool, lam: float = 1.0) -> None:
        """
        Layer 3: recency-weighted Beta-Bernoulli update.

            a' = lam * a + s
            b' = lam * b + (1 - s)

        lam = 1 gives the standard conjugate update; lam < 1 discounts
        old evidence (recency bias).
        """
        s = 1.0 if success else 0.0
        self.a = lam * self.a + s
        self.b = lam * self.b + (1.0 - s)

    # ----- Layer 1: cognitive antecedents -------------------------------

    def perceived_trustworthiness(self, alpha: float = 0.7,
                                  beta: float = 0.2,
                                  gamma: float = 0.1) -> float:
        """
        Layer 1: weighted sum of perceived ability, benevolence, integrity.

            P = (alpha*A + beta*B + gamma*I) / (alpha + beta + gamma)

        Falls back to the Beta mean for any trait that is not set.
        """
        fallback = self.bayes_mean()
        A = self.A if self.A is not None else fallback
        B = self.B if self.B is not None else fallback
        I = self.I if self.I is not None else fallback
        s = alpha + beta + gamma
        if s <= 0.0:
            return fallback
        return (alpha * A + beta * B + gamma * I) / s

    # ----- Layer 4: asymmetric temporal dynamics ------------------------

    def asymmetric_dynamics(self, success: bool,
                            k_build: Optional[float] = None,
                            k_destroy: Optional[float] = None,
                            dt: float = 1.0) -> float:
        """
        Layer 4: one Euler step of asymmetric build/destroy dynamics.

            On success:  T <- T + k_build * (1 - T) * dt
            On failure:  T <- T - k_destroy * T * dt

        k_destroy >> k_build captures the well-documented negativity bias.
        """
        kb = self.k_build if k_build is None else k_build
        kd = self.k_destroy if k_destroy is None else k_destroy
        t = self.trust if self.trust is not None else self.bayes_mean()
        if success:
            t = t + kb * (1.0 - t) * dt
        else:
            t = t - kd * t * dt
        t = min(max(t, 0.0), 1.0)
        self.trust = t
        return t

    def recovery_time(self, target: float = 0.8,
                      k_build: Optional[float] = None) -> float:
        """
        Closed-form recovery time from current trust to a target level
        under repeated success (continuous approximation):

            t = (1/k_build) * ln( (1 - T0) / (1 - T_target) )
        """
        kb = self.k_build if k_build is None else k_build
        if kb <= 0.0:
            return float("inf")
        t0 = self.trust if self.trust is not None else self.bayes_mean()
        t0 = min(max(t0, 0.0), 1.0 - 1e-12)
        target = min(max(target, t0), 1.0 - 1e-12)
        return (1.0 / kb) * math.log((1.0 - t0) / (1.0 - target))

    # ----- Convenience --------------------------------------------------

    def current_trust(self) -> float:
        """Return the explicit trust if set, otherwise the Beta mean."""
        return self.trust if self.trust is not None else self.bayes_mean()

    def reset(self) -> None:
        """Clear the explicit trust value."""
        self.trust = None


# ===========================================================================
# 4. Trustor — the agent making the decision
# ===========================================================================

@dataclass
class Trustor:
    """
    The trustor agent. Holds dispositional and risk parameters:

        trust_propensity (tau)  -- baseline willingness to trust
        risk_aversion    (rho)  -- threshold multiplier on stakes
        delta_star              -- repeated-game betrayal threshold
        beta_s                  -- logistic steepness for Layer 5
    """
    id: str = "trustor"
    trust_propensity: float = 0.5
    risk_aversion: float = 0.2
    delta_star: float = 0.708
    beta_s: float = 10.0


# ===========================================================================
# 5. TrustGroup — Layer 7: Group Trust Extension
# ===========================================================================

@dataclass
class TrustGroup:
    """
    A collective trustee composed of m individual agents.

    Aggregation architectures:
        SERIES   : min(T_j)                (weakest link)
        PARALLEL : 1 - prod(1 - T_j)       (redundancy)
        QUORUM   : P(X >= k), X ~ BetaBin  (correlated k-of-m)
        CUSTOM   : user-supplied monotone aggregator

    Cohesion-diversity (proposed heuristic):

        kappa = min{ 1, exp( -nu_pen * max(0, var - tau)
                             + nu_rew * H ) }

    Collective strategic factor:
        Phi_G = sigma( beta_s * (Delta_G - delta_star) )
    """
    members: List[Trustee]
    aggregation: AggregationType = AggregationType.SERIES

    # Quorum parameters
    quorum_k: Optional[int] = None
    correlation_rho: float = 0.0

    # Cohesion-diversity parameters
    cohesion_pen: float = 0.0
    cohesion_rew: float = 0.0
    cohesion_tau: float = 0.0
    entropy_override: Optional[float] = None

    # Custom aggregator
    custom_aggregator: Optional[Callable[[Sequence[float]], float]] = None

    # Collective strategic parameters
    collective_discount: Optional[float] = None
    apply_collective_strategic: bool = False
    delta_star: float = 0.708
    beta_s: float = 10.0

    # ------------------------------------------------------------------

    def individual_trusts(self) -> List[float]:
        """Vector of current individual trust scores."""
        return [m.current_trust() for m in self.members]

    # ----- 5.1 Aggregation ---------------------------------------------

    def aggregate_raw(self) -> float:
        """Apply the chosen aggregation function (without cohesion)."""
        ts = self.individual_trusts()
        m = len(ts)
        if m == 0:
            raise ValueError("TrustGroup must contain at least one member.")

        if self.aggregation == AggregationType.SERIES:
            return min(ts)

        if self.aggregation == AggregationType.PARALLEL:
            prod = 1.0
            for t in ts:
                prod *= (1.0 - t)
            return 1.0 - prod

        if self.aggregation == AggregationType.QUORUM:
            if self.quorum_k is None:
                raise ValueError("quorum_k must be set for QUORUM aggregation.")
            return self._quorum_probability(ts)

        if self.aggregation == AggregationType.CUSTOM:
            if self.custom_aggregator is None:
                raise ValueError(
                    "custom_aggregator must be provided for CUSTOM aggregation.")
            return float(self.custom_aggregator(ts))

        raise ValueError(f"Unknown aggregation type: {self.aggregation}")

    def _quorum_probability(self, ts: Sequence[float]) -> float:
        """P(at least k of m trustworthy) under beta-binomial correlation."""
        m = len(ts)
        k = int(self.quorum_k)  # type: ignore[arg-type]
        k = max(1, min(k, m))
        T_bar = sum(ts) / m

        rho = min(max(self.correlation_rho, 0.0), 0.999)
        if rho < 1e-6:
            # Independence limit -> ordinary binomial
            return _binomial_sf(k, m, T_bar)

        scale = (1.0 - rho) / rho
        alpha = max(T_bar * scale, 1e-3)
        beta = max((1.0 - T_bar) * scale, 1e-3)
        return _beta_binomial_sf(k, m, alpha, beta)

    # ----- 5.2 Cohesion-diversity --------------------------------------

    def cohesion_factor(self) -> float:
        """
        Proposed heuristic cohesion-diversity multiplier kappa.

        Returns 1.0 for a single-member group.
        """
        ts = self.individual_trusts()
        if len(ts) <= 1:
            return 1.0

        var = _variance(ts)

        if self.entropy_override is not None:
            H = self.entropy_override
        else:
            s = sum(ts)
            if s <= 0.0:
                H = 0.0
            else:
                H = _shannon_entropy([t / s for t in ts])

        excess = max(0.0, var - self.cohesion_tau)
        exponent = -self.cohesion_pen * excess + self.cohesion_rew * H
        # Guard against overflow before clipping
        if exponent > 700.0:
            return 1.0
        return min(1.0, math.exp(exponent))

    # ----- 5.3 Aggregate trust -----------------------------------------

    def aggregate_trust(self) -> float:
        """Raw aggregate trust multiplied by the cohesion factor."""
        raw = self.aggregate_raw()
        kappa = self.cohesion_factor()
        return min(max(raw * kappa, 0.0), 1.0)

    # ----- 5.4 Collective strategic factor -----------------------------

    def collective_strategic_factor(self,
                                    delta_star: Optional[float] = None) -> float:
        """
        Layer 7.4: Phi_G = sigma( beta_s * (Delta_G - delta_star) ).

        If collective_discount is not set, the mean of member discount
        factors is used.
        """
        ds = self.delta_star if delta_star is None else delta_star
        if self.collective_discount is not None:
            delta_G = self.collective_discount
        elif self.members:
            delta_G = sum(m.discount_factor for m in self.members) / len(self.members)
        else:
            delta_G = 0.0
        return sigmoid(self.beta_s * (delta_G - ds))

    # ----- 5.5 Effective group trust -----------------------------------

    def effective_trust(self) -> float:
        """
        Complete group trust:
            T_eff = aggregate_trust * Phi_G   (Phi_G applied optionally)
        """
        t = self.aggregate_trust()
        if self.apply_collective_strategic:
            t *= self.collective_strategic_factor()
        return min(max(t, 0.0), 1.0)


# ===========================================================================
# 6. TrustScenario — end-to-end decision pipeline
# ===========================================================================

@dataclass
class TrustScenario:
    """
    Runs the complete dyadic or group decision pipeline.

    For a dyadic target:
        T_eff = tau * P * Phi(delta_j)
        EU    = T_eff * U_win + (1 - T_eff) * U_loss
        theta = rho * S
        decision = EU >= theta

    For a group target:
        T_eff = group.effective_trust()
        (same EU and threshold logic)
    """
    name: str
    trustor: Trustor
    target: object  # Trustee or TrustGroup
    U_win: float = 1.0
    U_loss: float = -1.0
    stakes: Optional[float] = None
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1

    # ------------------------------------------------------------------

    def _dyadic_trust(self, t: Trustee) -> float:
        # Cognitive antecedent (Layer 1) or explicit trust
        if t.trust is not None:
            p = t.trust
        else:
            p = t.perceived_trustworthiness(self.alpha, self.beta, self.gamma)
        # Propensity
        base = self.trustor.trust_propensity * p
        # Strategic smooth override (Layer 5)
        phi = strategic_factor(t.discount_factor,
                               self.trustor.delta_star,
                               self.trustor.beta_s)
        return min(max(base * phi, 0.0), 1.0)

    def target_trust(self) -> float:
        if isinstance(self.target, TrustGroup):
            return self.target.effective_trust()
        if isinstance(self.target, Trustee):
            return self._dyadic_trust(self.target)
        raise TypeError("target must be a Trustee or TrustGroup.")

    def expected_utility(self, trust: float) -> float:
        return trust * self.U_win + (1.0 - trust) * self.U_loss

    def threshold(self) -> float:
        s = self.stakes if self.stakes is not None else abs(self.U_loss)
        return self.trustor.risk_aversion * s

    def run(self) -> Tuple[bool, float, float]:
        """Return (decision, effective_trust, expected_utility)."""
        trust = self.target_trust()
        eu = self.expected_utility(trust)
        theta = self.threshold()
        return (eu >= theta), trust, eu

    def report(self) -> Dict[str, float]:
        """Return a human-readable dictionary with all intermediate values."""
        trust = self.target_trust()
        eu = self.expected_utility(trust)
        theta = self.threshold()
        return {
            "name": self.name,
            "effective_trust": trust,
            "expected_utility": eu,
            "threshold": theta,
            "decision": eu >= theta,
        }


# ===========================================================================
# 7. TrustNetwork — Layer 6: social propagation
# ===========================================================================

class TrustNetwork:
    """
    Layer 6: social network propagation of trust.

        T_indirect[i,j] = sum_k T[i,k] * T[k,j] * phi[i,k]
                          --------------------------------
                          sum_k T[i,k] * phi[i,k] + eps

        T(t+1) = mu * T_direct(t+1) + (1 - mu) * T_indirect(t)
    """

    def __init__(self, n_agents: int):
        self.n = n_agents
        self.direct: List[List[float]] = [[0.0] * n_agents for _ in range(n_agents)]
        self.indirect: List[List[float]] = [[0.0] * n_agents for _ in range(n_agents)]
        self.recommender_credibility: List[List[float]] = [
            [1.0] * n_agents for _ in range(n_agents)
        ]

    def set_direct(self, i: int, j: int, value: float) -> None:
        self.direct[i][j] = min(max(value, 0.0), 1.0)

    def set_credibility(self, i: int, k: int, value: float) -> None:
        self.recommender_credibility[i][k] = min(max(value, 0.0), 1.0)

    def compute_indirect(self, epsilon: float = 1e-9) -> None:
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.indirect[i][j] = 0.0
                    continue
                num = 0.0
                den = epsilon
                for k in range(self.n):
                    if k == i or k == j:
                        continue
                    w = self.direct[i][k] * self.recommender_credibility[i][k]
                    num += w * self.direct[k][j]
                    den += w
                self.indirect[i][j] = num / den

    def update(self, mu: float = 0.5) -> List[List[float]]:
        """One global damping update. Returns the new trust matrix."""
        self.compute_indirect()
        new = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                new[i][j] = (mu * self.direct[i][j]
                             + (1.0 - mu) * self.indirect[i][j])
        self.direct = new
        return new

    def iterate(self, steps: int = 10, mu: float = 0.5) -> List[List[float]]:
        for _ in range(steps):
            self.update(mu=mu)
        return self.direct


# ===========================================================================
# 8. Simulation study (proof-of-concept, Section 6)
# ===========================================================================

def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def simulate_comparison(n_scenarios: int = 2000,
                        seed: int = 0,
                        rho_true: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Reproduce the proof-of-concept simulation study of Section 6.

    Ground truth is generated from a *separate* process that does NOT
    assume the proposed model's aggregation rules. We compare four
    decision models:

        - "unified"  : the proposed framework
        - "average"  : simple mean of individual trust scores
        - "minmax"   : min for series, max for parallel
        - "binomial" : independence-only quorum (no correlation/cohesion)

    Returns a dict of {model_name: {"accuracy": ..., "brier": ...}}.
    """
    rng = random.Random(seed)

    models = ["unified", "average", "minmax", "binomial"]
    acc = {m: 0 for m in models}
    brier = {m: 0.0 for m in models}
    counts = {m: 0 for m in models}

    for _ in range(n_scenarios):
        arch = rng.choice(["series", "parallel", "quorum"])
        m = rng.randint(2, 9)

        # True reliabilities drawn from a mixture of Beta distributions
        true_T = [rng.betavariate(2.0 + 6.0 * rng.random(),
                                  2.0 + 6.0 * (1 - rng.random()))
                  for _ in range(m)]

        # Latent factor for correlation
        latent = rng.random()
        corr_noise = [min(max(0.7 * t + 0.3 * latent + 0.05 * rng.gauss(0, 1),
                              0.0), 1.0) for t in true_T]

        # ---- Ground truth (independent of the proposed model) ---------
        if arch == "series":
            y = 1 if all(t >= 0.7 for t in corr_noise) else 0
        elif arch == "parallel":
            y = 1 if any(t >= 0.8 for t in corr_noise) else 0
        else:
            k = max(2, m // 2 + 1)
            y = 1 if sum(1 for t in corr_noise if t >= 0.75) >= k else 0

        # ---- Model predictions ----------------------------------------
        trusts = [rng.betavariate(max(1.0, 10 * t), max(1.0, 10 * (1 - t)))
                  for t in true_T]

        # Unified
        members = [Trustee(f"T{i}", a=10 * t, b=10 * (1 - t))
                   for i, t in enumerate(trusts)]
        if arch == "series":
            g = TrustGroup(members, aggregation=AggregationType.SERIES,
                           cohesion_pen=5.0, cohesion_tau=0.02)
        elif arch == "parallel":
            g = TrustGroup(members, aggregation=AggregationType.PARALLEL,
                           cohesion_pen=2.0, cohesion_tau=0.05)
        else:
            k = max(2, m // 2 + 1)
            g = TrustGroup(members, aggregation=AggregationType.QUORUM,
                           quorum_k=k, correlation_rho=rho_true,
                           cohesion_pen=2.0, cohesion_tau=0.02)
        p_unified = g.aggregate_trust()

        # Simple average
        p_average = sum(trusts) / m

        # Min/Max heuristic
        if arch == "series":
            p_minmax = min(trusts)
        elif arch == "parallel":
            p_minmax = max(trusts)
        else:
            k = max(2, m // 2 + 1)
            p_minmax = _binomial_sf(k, m, sum(trusts) / m)

        # Naive binomial quorum
        k = max(2, m // 2 + 1)
        p_binomial = _binomial_sf(k, m, sum(trusts) / m)

        preds = {
            "unified": p_unified,
            "average": p_average,
            "minmax": p_minmax,
            "binomial": p_binomial,
        }

        for name, p in preds.items():
            pred = 1 if p >= 0.5 else 0
            acc[name] += int(pred == y)
            brier[name] += _brier(p, y)
            counts[name] += 1

    results: Dict[str, Dict[str, float]] = {}
    for name in models:
        results[name] = {
            "accuracy": acc[name] / max(counts[name], 1),
            "brier": brier[name] / max(counts[name], 1),
            "n": counts[name],
        }
    return results


# ===========================================================================
# 9. Demo (run as `python trustlib.py`)
# ===========================================================================

def _demo() -> None:  # pragma: no cover
    print(f"trustlib v{__version__} — demo of the five paper case studies\n")

    # Case 1: Autonomous vehicle
    passenger = Trustor("passenger", trust_propensity=0.7, risk_aversion=0.15)
    car = Trustee("car", a=10, b=1, discount_factor=0.9)
    car.trust = 0.65
    car.asymmetric_dynamics(success=False, k_build=0.1, k_destroy=0.8)
    sc = TrustScenario("Robo-taxi", passenger, car,
                       U_win=50, U_loss=-200, stakes=100)
    print("Case 1 (AV):", sc.report())

    # Case 2: P2P lending
    lender = Trustor("lender", trust_propensity=0.5, risk_aversion=0.25)
    borrower = Trustee("borrower", a=68, b=2, discount_factor=0.6)
    sc = TrustScenario("P2P Loan", lender, borrower,
                       U_win=750, U_loss=-5000, stakes=5000)
    print("Case 2 (P2P):", sc.report())

    # Case 3: Blockchain validator
    delegator = Trustor("delegator", trust_propensity=0.8, risk_aversion=0.1)
    validator = Trustee("validator", a=95, b=5, discount_factor=0.9)
    sc = TrustScenario("Validator", delegator, validator,
                       U_win=5000, U_loss=-10000, stakes=10000)
    print("Case 3 (Validator):", sc.report())

    # Case 4: Autonomous truck platoon
    trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
    trucks.append(Trustee("T4", a=40, b=60))
    platoon = TrustGroup(trucks, aggregation=AggregationType.SERIES,
                         cohesion_pen=10.0, cohesion_tau=0.02)
    print("Case 4 (Platoon) T_agg =", round(platoon.aggregate_trust(), 3))

    # Case 5: Hybrid fact-checking
    ai1 = Trustee("AI1", a=80, b=20)
    ai2 = Trustee("AI2", a=80, b=20)
    ai3 = Trustee("AI3", a=70, b=30)
    hum1 = Trustee("Hum1", a=90, b=10)
    hum2 = Trustee("Hum2", a=85, b=15)
    collective = TrustGroup([ai1, ai2, ai3, hum1, hum2],
                            aggregation=AggregationType.QUORUM,
                            quorum_k=3,
                            cohesion_pen=20.0, cohesion_tau=0.1,
                            cohesion_rew=5.0,
                            entropy_override=0.673)
    print("Case 5 (Fact-check) T_agg =", round(collective.aggregate_trust(), 3))


if __name__ == "__main__":  # pragma: no cover
    _demo()
    