"""Minimal two-bus AC power-flow PUBO for small-qubit experiments.

Run from repository root:
    python examples/powerflow/powerflow_2bus.py --bits 2 --num-reads 200

Network: 100 MVA base; slack V0=1+0j; PQ load P=0.5, Q=0.2 p.u.;
single series line Z=0.02+0.10j p.u., without shunt charging.

The existing PowerFlow.generate_discretization() currently hardcodes
mu in [0.8, 1.2] and omega in [-0.2, 0.2]. --bits controls resolution;
changing bounds requires editing the PowerFlow implementation itself.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engioptiqa import AnnealingSolverOpenJij, PowerFlow, PowerFlowData


def main():
    parser = argparse.ArgumentParser(description="Two-bus one-shot power-flow PUBO (OpenJij)")
    parser.add_argument("--bits", type=int, default=2,
                        help="Binary variables per voltage component (default: 2)")
    parser.add_argument("--num-reads", type=int, default=200,
                        help="OpenJij simulated-annealing reads (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bits < 1 or args.num_reads < 1:
        parser.error("--bits and --num-reads must be positive")

    case_dir = Path(__file__).resolve().parent / "data" / "2-bus"
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Missing two-bus data directory: {case_dir}")

    data = PowerFlowData(str(case_dir))
    if data.n_bus != 2 or data.ref_bus != 0 or data.pq_buses != [1]:
        raise ValueError("Expected slack bus 0 and PQ load bus 1")

    problem = PowerFlow(data)
    problem.generate_discretization(
        n_qubits_per_var=args.bits,
        binary_representation="adaptive_range",
        lower_lim=-0.8,
        upper_lim=1.2,
    )
    problem.generate_problem_formulation()

    n_variables = len(problem.binary_model.get_variables())
    expected = 2 * args.bits  # Only mu_1 and omega_1 are unknown.
    print(f"Logical binary variables: {n_variables} (expected {expected})")
    if n_variables != expected:
        raise RuntimeError("Unexpected binary variable count")

    solver = AnnealingSolverOpenJij()
    solver.setup_solver()
    solver.solve_problem(problem, num_reads=args.num_reads, seed=args.seed)

    best = None
    for sample in problem.results:
        mu, omega = problem.decode_voltages(sample)
        residual = problem.compute_residual(mu, omega)
        if np.isfinite(residual) and (best is None or residual < best[0]):
            best = (residual, mu.copy(), omega.copy())
    if best is None:
        raise RuntimeError("No finite solution returned by OpenJij")

    residual, mu, omega = best
    voltage, angle = problem.voltage_observables(mu, omega)
    print("Bus 0 (slack): V = 1.000000 + 0.000000j")
    print(f"Bus 1 (PQ):    V = {mu[1]:.6f} {omega[1]:+.6f}j")
    print(f"Bus 1: |V| = {voltage[1]:.6f} p.u.; angle = {angle[1]:.6f} deg")
    print(f"Best physical residual = {residual:.6e}")
    print("Continuous high-voltage benchmark: V1 ≈ 0.96678046 - 0.04600000j")


if __name__ == "__main__":
    main()
