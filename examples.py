"""
examples.py — Reproduces the five case studies from the paper.
Run: python examples.py
"""

from trustlib import (
    Trustee, Trustor, TrustGroup, TrustScenario,
    AggregationType, harmonic_mean, simulate_comparison,
)


def case_1_autonomous_vehicle():
    print("\n=== Case 1: Autonomous Vehicle (Dyadic) ===")
    passenger = Trustor("passenger", trust_propensity=0.7, risk_aversion=0.15)
    car = Trustee("car", a=10, b=1, discount_factor=0.9)

    # Set explicit initial trust and apply asymmetric failure dynamics
    car.trust = 0.65
    car.asymmetric_dynamics(success=False, k_build=0.1, k_destroy=0.8)
    print(f"Trust after failure: {car.trust:.3f}")
    print(f"Recovery time to 0.8: {car.recovery_time(0.8):.2f} rides")

    sc = TrustScenario("Robo-taxi", passenger, car,
                       U_win=50, U_loss=-200, stakes=100)
    print(sc.report())


def case_2_p2p_lending():
    print("\n=== Case 2: P2P Lending (Dyadic + Strategic) ===")
    lender = Trustor("lender", trust_propensity=0.5, risk_aversion=0.25)
    borrower = Trustee("borrower", a=68, b=2, discount_factor=0.6)
    sc = TrustScenario("P2P Loan", lender, borrower,
                       U_win=750, U_loss=-5000, stakes=5000)
    print(sc.report())


def case_3_blockchain_validator():
    print("\n=== Case 3: Blockchain Validator (Dyadic) ===")
    delegator = Trustor("delegator", trust_propensity=0.8, risk_aversion=0.1)
    validator = Trustee("validator", a=95, b=5, discount_factor=0.9)
    sc = TrustScenario("Validator", delegator, validator,
                       U_win=5000, U_loss=-10000, stakes=10000)
    print(sc.report())


def case_4_platoon():
    print("\n=== Case 4: Autonomous Truck Platoon (Series) ===")
    trucks = [Trustee(f"T{i}", a=92, b=8) for i in range(4)]
    trucks.append(Trustee("T4", a=40, b=60))
    platoon = TrustGroup(trucks, aggregation=AggregationType.SERIES,
                         cohesion_pen=10.0, cohesion_tau=0.02)
    print(f"Individual trusts: {[round(t.current_trust(), 3) for t in trucks]}")
    print(f"Cohesion factor:   {platoon.cohesion_factor():.3f}")
    print(f"Aggregate trust:   {platoon.aggregate_trust():.3f}")

    # Decision
    manager = Trustor("manager", trust_propensity=1.0, risk_aversion=0.1)
    sc = TrustScenario("Platoon", manager, platoon,
                       U_win=10_000, U_loss=-500_000, stakes=500_000)
    print(sc.report())


def case_5_fact_checking():
    print("\n=== Case 5: Hybrid Fact-Checking (Quorum + Diversity) ===")
    members = [
        Trustee("AI1", a=80, b=20),
        Trustee("AI2", a=80, b=20),
        Trustee("AI3", a=70, b=30),
        Trustee("Hum1", a=90, b=10),
        Trustee("Hum2", a=85, b=15),
    ]
    collective = TrustGroup(
        members,
        aggregation=AggregationType.QUORUM,
        quorum_k=3,
        cohesion_pen=20.0,
        cohesion_tau=0.1,
        cohesion_rew=5.0,
        entropy_override=0.673,   # label entropy: 2 True / 3 False
    )
    print(f"Cohesion factor: {collective.cohesion_factor():.3f}")
    print(f"Aggregate trust: {collective.aggregate_trust():.3f}")


def case_6_network_propagation():
    print("\n=== Case 6: Network Propagation (Layer 6) ===")
    from trustlib import TrustNetwork
    net = TrustNetwork(4)
    # Direct trust: agent 0 trusts agent 1 strongly, agent 1 trusts 2, etc.
    net.set_direct(0, 1, 0.9)
    net.set_direct(0, 2, 0.4)
    net.set_direct(1, 2, 0.95)
    net.set_direct(1, 3, 0.7)
    net.set_direct(2, 3, 0.85)
    net.iterate(steps=3, mu=0.5)
    print("Direct trust matrix after propagation:")
    for row in net.direct:
        print("  ", [round(x, 3) for x in row])


def case_7_simulation():
    print("\n=== Simulation Study (proof-of-concept) ===")
    results = simulate_comparison(n_scenarios=2000, seed=42)
    print(f"{'Model':<12}{'Accuracy':>12}{'Brier':>12}")
    for name, r in results.items():
        print(f"{name:<12}{r['accuracy']:>12.4f}{r['brier']:>12.4f}")


if __name__ == "__main__":
    case_1_autonomous_vehicle()
    case_2_p2p_lending()
    case_3_blockchain_validator()
    case_4_platoon()
    case_5_fact_checking()
    case_6_network_propagation()
    case_7_simulation()
    