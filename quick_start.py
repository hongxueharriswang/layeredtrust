from trustlib import Trustee, Trustor, TrustGroup, TrustScenario, AggregationType

# --- Dyadic ---
buyer  = Trustor("buyer", trust_propensity=0.6, risk_aversion=0.2)
seller = Trustee("seller", a=130, b=20, discount_factor=0.8)
sc = TrustScenario("Amazon", buyer, seller, U_win=100, U_loss=-800, stakes=800)
print(sc.report())
# {'name': 'Amazon', 'effective_trust': 0.372, 'expected_utility': -465.2,
#  'threshold': 160.0, 'decision': False}

# --- Group (series) ---
trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
trucks.append(Trustee("T4", a=40, b=60))
platoon = TrustGroup(trucks, aggregation=AggregationType.SERIES,
                     cohesion_pen=10.0, cohesion_tau=0.02)
print(platoon.aggregate_trust())   # ~0.317

# --- Group (quorum with correlation) ---
signers = [Trustee(f"S{i}", a=90, b=10) for i in range(6)]
signers.append(Trustee("M", a=60, b=40))
signers.extend([Trustee(f"U{i}", a=40, b=60) for i in range(2)])
dao = TrustGroup(signers, aggregation=AggregationType.QUORUM,
                 quorum_k=5, correlation_rho=0.2,
                 cohesion_pen=2.0, cohesion_tau=0.02)
print(dao.aggregate_trust())       # ~0.87

# --- Simulation study ---
from trustlib import simulate_comparison
results = simulate_comparison(n_scenarios=2000, seed=42)
for name, r in results.items():
    print(f"{name:<10} acc={r['accuracy']:.4f}  brier={r['brier']:.4f}")
    