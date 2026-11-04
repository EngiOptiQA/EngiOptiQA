
"""Iterative AQPF with two binary variables per voltage component.

EngiOptiQA formulates each PUBO.
OpenJij solves each PUBO.
This script controls the iterative voltage updates.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


# Repository root, relative to examples/powerflow/iterative_aqpf.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engioptiqa import (
    AnnealingSolverOpenJij,
    PowerFlow,
    PowerFlowData,
)


def solve_iterative(
    problem,
    num_reads=200,
    max_iter=100,
    tolerance=1e-4,
    seed=0,
    initial_delta_mu=0.04,
    initial_delta_omega=0.02,
    min_delta=1e-5,
    max_delta=0.5,
    shrink_factor=0.8,
    grow_factor=1.5,
):
    """Repeatedly solve local two-bit PUBOs around current voltages."""

    n_bus = problem.data.n_bus
    slack = problem.data.ref_bus

    # Initial flat voltage profile.
    mu = np.ones(n_bus, dtype=float)
    omega = np.zeros(n_bus, dtype=float)

    residual = problem.compute_residual(mu, omega)

    # Create the sampler only once.
    solver = AnnealingSolverOpenJij()
    solver.setup_solver()

    delta_mu = initial_delta_mu
    delta_omega = initial_delta_omega

    expected_bits = 4 * (n_bus - 1)

    history = [residual]
    start_time = perf_counter()

    print("\n=== Iterative AQPF with OpenJij ===")
    print(f"Buses: {n_bus}")
    print("Bits per voltage component: 2")
    print(f"Expected binary variables: {expected_bits}")
    print(f"Initial residual: {residual:.6e}")

    for iteration in range(max_iter):

        # Stop before creating another optimization problem.
        if residual <= tolerance:
            print("\nConvergence tolerance reached.")
            break

        # 1. Construct a fresh local binary representation.
        problem.generate_aqpf_discretization(
            mu_center=mu,
            omega_center=omega,
            delta_mu=delta_mu,
            delta_omega=delta_omega,
        )

        # 2. Generate the physical power-flow PUBO.
        problem.generate_problem_formulation()

        actual_bits = len(
            problem.binary_model.get_variables()
        )

        if actual_bits != expected_bits:
            raise RuntimeError(
                f"Expected {expected_bits} binary variables, "
                f"but formulation contains {actual_bits}."
            )

        # 3. Solve the PUBO using OpenJij.
        solver.solve_problem(
            problem,
            num_reads=num_reads,
            seed=seed + iteration,
        )

        # 4. Select the sample with the best physical residual.
        # Start with the previous best to prevent regression.
        best_mu = mu
        best_omega = omega
        best_residual = residual

        sample_count = 0

        for sample in problem.results:
            sample_count += 1

            candidate_mu, candidate_omega = (
                problem.decode_voltages(sample)
            )

            # The slack bus is fixed at 1 + 0j.
            candidate_mu[slack] = 1.0
            candidate_omega[slack] = 0.0

            if (
                not np.all(np.isfinite(candidate_mu))
                or not np.all(np.isfinite(candidate_omega))
            ):
                continue

            candidate_residual = problem.compute_residual(
                candidate_mu,
                candidate_omega,
            )

            if (
                np.isfinite(candidate_residual)
                and candidate_residual < best_residual
            ):
                best_residual = candidate_residual
                best_mu = candidate_mu.copy()
                best_omega = candidate_omega.copy()

        if sample_count == 0:
            raise RuntimeError(
                "OpenJij returned no samples."
            )

        # 5. Update the voltage center and the search increments.
        improvement = best_residual < residual

        if improvement:
            mu = best_mu
            omega = best_omega
            residual = best_residual

            # Refine around a successfully improved center.
            delta_mu = max(
                min_delta,
                delta_mu * shrink_factor,
            )
            delta_omega = max(
                min_delta,
                delta_omega * shrink_factor,
            )

        else:
            # Explore a larger neighborhood if no improvement.
            delta_mu = min(
                max_delta,
                delta_mu * grow_factor,
            )
            delta_omega = min(
                max_delta,
                delta_omega * grow_factor,
            )

        history.append(residual)

        print(
            f"Iter {iteration + 1:3d} | "
            f"residual={residual:.6e} | "
            f"bits={actual_bits} | "
            f"d_mu={delta_mu:.5g} | "
            f"d_omega={delta_omega:.5g} | "
            f"{'improved' if improvement else 'no improvement'}"
        )

    elapsed = perf_counter() - start_time

    voltage, angle = problem.voltage_observables(
        mu, omega
    )

    print("\n=== Final AQPF result ===")
    print("mu:", np.round(mu, 6))
    print("omega:", np.round(omega, 6))
    print("|V| (p.u.):", np.round(voltage, 6))
    print("theta (deg):", np.round(angle, 6))
    print(f"Final residual: {residual:.6e}")
    print(f"Tolerance: {tolerance:.6e}")
    print(f"Converged: {residual <= tolerance}")
    print(f"Iterations: {len(history) - 1}")
    print(f"Total elapsed time: {elapsed:.2f} s")

    return mu, omega, np.asarray(history)


def main():
    parser = argparse.ArgumentParser(
        description="Iterative two-bit AQPF using OpenJij"
    )

    parser.add_argument(
        "--bus",
        type=int,
        default=4,
        choices=[2, 4, 9, 14],
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--delta-mu",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--delta-omega",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    if args.num_reads < 1 or args.max_iter < 1:
        parser.error(
            "num-reads and max-iter must be positive"
        )

    if (
        not np.isfinite(args.tolerance)
        or args.tolerance <= 0
        or not np.isfinite(args.delta_mu)
        or args.delta_mu <= 0
        or not np.isfinite(args.delta_omega)
        or args.delta_omega <= 0
    ):
        parser.error(
            "Tolerance and increments must be positive and finite"
        )

    case_path = (
        ROOT
        / "examples"
        / "powerflow"
        / "data"
        / f"{args.bus}-bus"
    )

    if not case_path.is_dir():
        raise FileNotFoundError(
            f"Case data not found: {case_path}"
        )

    data = PowerFlowData(str(case_path))
    problem = PowerFlow(data)

    mu, omega, history = solve_iterative(
        problem=problem,
        num_reads=args.num_reads,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        seed=args.seed,
        initial_delta_mu=args.delta_mu,
        initial_delta_omega=args.delta_omega,
    )

    if (
        not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(omega))
        or not np.all(np.isfinite(history))
    ):
        raise RuntimeError(
            "AQPF returned non-finite results."
        )

    if np.any(np.diff(history) > 1e-12):
        raise RuntimeError(
            "Residual increased between iterations."
        )


if __name__ == "__main__":
    main()
