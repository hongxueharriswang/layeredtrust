"""
test_trustlib.py — Unit tests for the trustlib library.
Run: python -m unittest test_trustlib -v
"""

import math
import unittest

from trustlib import (
    Trustee, Trustor, TrustGroup, TrustScenario, TrustNetwork,
    AggregationType, harmonic_mean, geometric_mean,
    sigmoid, strategic_factor,
)
from trustlib import _binomial_sf, _beta_binomial_sf


class TestMathHelpers(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(10.0), 0.99)
        self.assertLess(sigmoid(-10.0), 0.01)

    def test_strategic_factor_limits(self):
        # Far above threshold -> close to 1
        self.assertGreater(strategic_factor(0.9, 0.708, 20.0), 0.97)
        # Far below threshold -> close to 0
        self.assertLess(strategic_factor(0.5, 0.708, 20.0), 0.03)
        # At threshold -> 0.5
        self.assertAlmostEqual(strategic_factor(0.708, 0.708, 10.0), 0.5)

    def test_binomial_sf(self):
        # Fair coin, 10 flips, P(X >= 5) = 0.623...
        self.assertAlmostEqual(_binomial_sf(5, 10, 0.5), 0.6230, places=3)
        # All fail
        self.assertAlmostEqual(_binomial_sf(0, 5, 0.0), 1.0)
        # Impossible
        self.assertAlmostEqual(_binomial_sf(6, 5, 0.9), 0.0)

    def test_beta_binomial_sf_rho_zero(self):
        # With rho -> 0 the beta-binomial must approach the binomial
        p = 0.7
        m, k = 9, 5
        binom = _binomial_sf(k, m, p)
        # Build beta-binomial with tiny rho
        rho = 1e-4
        scale = (1 - rho) / rho
        alpha = p * scale
        beta = (1 - p) * scale
        bb = _beta_binomial_sf(k, m, alpha, beta)
        self.assertAlmostEqual(binom, bb, places=4)


class TestTrustee(unittest.TestCase):
    def test_bayes_mean(self):
        t = Trustee("t", a=3, b=1)
        self.assertAlmostEqual(t.bayes_mean(), 0.75)

    def test_bayes_update(self):
        t = Trustee("t", a=1, b=1)
        t.update_bayes(success=True)
        self.assertEqual(t.a, 2.0)
        self.assertEqual(t.b, 1.0)
        t.update_bayes(success=False)
        self.assertEqual(t.a, 2.0)
        self.assertEqual(t.b, 2.0)

    def test_recency_update(self):
        t = Trustee("t", a=10, b=10)
        t.update_bayes(success=True, lam=0.5)
        self.assertAlmostEqual(t.a, 6.0)
        self.assertAlmostEqual(t.b, 5.0)

    def test_asymmetric_dynamics(self):
        t = Trustee("t")
        t.trust = 0.65
        t.asymmetric_dynamics(success=False, k_build=0.1, k_destroy=0.8)
        self.assertAlmostEqual(t.trust, 0.13, places=3)
        t.asymmetric_dynamics(success=True, k_build=0.1, k_destroy=0.8)
        self.assertAlmostEqual(t.trust, 0.13 + 0.1 * (1 - 0.13), places=3)

    def test_recovery_time(self):
        t = Trustee("t")
        t.trust = 0.13
        rt = t.recovery_time(target=0.8, k_build=0.1)
        expected = 10 * math.log((1 - 0.13) / (1 - 0.8))
        self.assertAlmostEqual(rt, expected, places=3)

    def test_bounds(self):
        t = Trustee("t", a=1, b=1)
        for _ in range(100):
            t.asymmetric_dynamics(success=True, k_build=0.5, k_destroy=0.5)
        self.assertLessEqual(t.trust, 1.0)
        self.assertGreaterEqual(t.trust, 0.0)


class TestTrustGroup(unittest.TestCase):
    def _members(self, vals):
        return [Trustee(f"T{i}", a=10 * v, b=10 * (1 - v))
                for i, v in enumerate(vals)]

    def test_series(self):
        g = TrustGroup(self._members([0.9, 0.8, 0.7]),
                       aggregation=AggregationType.SERIES)
        self.assertAlmostEqual(g.aggregate_raw(), 0.7, places=6)

    def test_parallel(self):
        g = TrustGroup(self._members([0.5, 0.5, 0.5]),
                       aggregation=AggregationType.PARALLEL)
        # 1 - 0.5^3 = 0.875
        self.assertAlmostEqual(g.aggregate_raw(), 0.875, places=6)

    def test_quorum_independent(self):
        g = TrustGroup(self._members([0.5] * 5),
                       aggregation=AggregationType.QUORUM,
                       quorum_k=3, correlation_rho=0.0)
        # Binomial(5, 0.5), P(X >= 3) = 0.5
        self.assertAlmostEqual(g.aggregate_raw(), 0.5, places=4)

    def test_quorum_correlated(self):
        # Positive correlation increases variance -> the SF at k = m/2+1
        # should move toward the tails. For p=0.5 exactly symmetric, SF at
        # k=3 should stay near 0.5, but for p>0.5 correlation reduces it.
        g_ind = TrustGroup(self._members([0.8] * 5),
                           aggregation=AggregationType.QUORUM,
                           quorum_k=4, correlation_rho=0.0)
        g_corr = TrustGroup(self._members([0.8] * 5),
                            aggregation=AggregationType.QUORUM,
                            quorum_k=4, correlation_rho=0.3)
        self.assertGreater(g_ind.aggregate_raw(), g_corr.aggregate_raw())

    def test_custom_harmonic(self):
        g = TrustGroup(self._members([0.9, 0.5, 0.9]),
                       aggregation=AggregationType.CUSTOM,
                       custom_aggregator=harmonic_mean)
        expected = harmonic_mean([0.9, 0.5, 0.9])
        self.assertAlmostEqual(g.aggregate_raw(), expected, places=6)

    def test_cohesion_single_member(self):
        g = TrustGroup(self._members([0.8]))
        self.assertEqual(g.cohesion_factor(), 1.0)

    def test_cohesion_penalty(self):
        g = TrustGroup(self._members([0.9, 0.9, 0.1]),
                       aggregation=AggregationType.SERIES,
                       cohesion_pen=10.0, cohesion_tau=0.0)
        self.assertLess(g.cohesion_factor(), 1.0)

    def test_cohesion_reward(self):
        g = TrustGroup(self._members([0.5, 0.5, 0.5]),
                       aggregation=AggregationType.PARALLEL,
                       cohesion_rew=2.0)
        # Entropy reward may push kappa above 1, clipped to 1
        self.assertLessEqual(g.cohesion_factor(), 1.0)

    def test_empty_group_raises(self):
        g = TrustGroup([], aggregation=AggregationType.SERIES)
        with self.assertRaises(ValueError):
            g.aggregate_raw()

    def test_quorum_missing_k_raises(self):
        g = TrustGroup(self._members([0.5] * 5),
                       aggregation=AggregationType.QUORUM)
        with self.assertRaises(ValueError):
            g.aggregate_raw()

    def test_bounds(self):
        g = TrustGroup(self._members([0.9, 0.1, 0.5]),
                       aggregation=AggregationType.PARALLEL,
                       cohesion_pen=5.0, cohesion_tau=0.0)
        t = g.aggregate_trust()
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)


class TestUnification(unittest.TestCase):
    """Check the Unification Theorem: m=1 reduces to the dyadic model."""

    def test_m_equals_1(self):
        # Dyadic
        t = Trustee("t", a=8, b=2, discount_factor=0.85)
        trustor = Trustor("u", trust_propensity=0.7, risk_aversion=0.2)
        dyadic = TrustScenario("d", trustor, t,
                               U_win=100, U_loss=-200, stakes=200)
        _, trust_d, eu_d = dyadic.run()

        # Group of one
        g = TrustGroup([t], aggregation=AggregationType.SERIES)
        # For m=1, aggregate_raw = t.trust = 0.8, kappa = 1
        self.assertAlmostEqual(g.aggregate_raw(), t.bayes_mean(), places=6)
        self.assertEqual(g.cohesion_factor(), 1.0)
        self.assertAlmostEqual(g.aggregate_trust(), t.bayes_mean(), places=6)

    def test_unification_bounds(self):
        t = Trustee("t", a=5, b=5)
        g = TrustGroup([t], aggregation=AggregationType.SERIES)
        self.assertGreaterEqual(g.aggregate_trust(), 0.0)
        self.assertLessEqual(g.aggregate_trust(), 1.0)


class TestTrustScenario(unittest.TestCase):
    def test_dyadic_decision_no(self):
        buyer = Trustor("buyer", trust_propensity=0.6, risk_aversion=0.2)
        seller = Trustee("seller", a=130, b=20, discount_factor=0.8)
        sc = TrustScenario("Amazon", buyer, seller,
                           U_win=100, U_loss=-800, stakes=800)
        decision, trust, eu = sc.run()
        self.assertFalse(decision)
        self.assertAlmostEqual(trust, 0.372, places=2)
        self.assertLess(eu, 0)

    def test_dyadic_decision_yes(self):
        # A very trustworthy trustee with a favorable payoff
        buyer = Trustor("buyer", trust_propensity=0.9, risk_aversion=0.05)
        seller = Trustee("seller", a=99, b=1, discount_factor=0.95)
        sc = TrustScenario("Amazon", buyer, seller,
                           U_win=1000, U_loss=-100, stakes=100)
        decision, trust, eu = sc.run()
        self.assertTrue(decision)
        self.assertGreater(eu, 0)

    def test_threshold(self):
        trustor = Trustor("t", risk_aversion=0.2)
        sc = TrustScenario("s", trustor, Trustee("x"),
                           U_win=1, U_loss=-1, stakes=500)
        self.assertAlmostEqual(sc.threshold(), 100.0)


class TestTrustNetwork(unittest.TestCase):
    def test_propagation(self):
        net = TrustNetwork(3)
        net.set_direct(0, 1, 0.9)
        net.set_direct(1, 2, 0.8)
        net.iterate(steps=5, mu=0.5)
        # After propagation, indirect trust from 0 to 2 should be positive
        self.assertGreater(net.direct[0][2], 0.0)
        # Bounds
        for row in net.direct:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


class TestAggregationHelpers(unittest.TestCase):
    def test_harmonic_mean(self):
        self.assertAlmostEqual(harmonic_mean([0.5, 0.5]), 0.5, places=6)
        self.assertLess(harmonic_mean([0.9, 0.1]), 0.5)

    def test_geometric_mean(self):
        self.assertAlmostEqual(geometric_mean([0.25, 0.25]), 0.25, places=6)
        self.assertLess(geometric_mean([0.9, 0.1]), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
    