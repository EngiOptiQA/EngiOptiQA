
"""EngiOptiQA power flow example using one-shot OpenJij PUBO solving."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engioptiqa import AnnealingSolverOpenJij, PowerFlow, PowerFlowData


def report(problem, mu, omega, title, tolerance=1e-4):
    """Print results and compare with the reference solution."""
    voltage, angles = problem.voltage_observables(mu, omega)
    residual = problem.compute_residual(mu, omega)

    print(f"\n{title}")
    print("mu:", np.round(mu, 6))
    print("omega:", np.round(omega, 6))
    print("|V| (p.u.):", np.round(voltage, 6))
    print("theta (deg):", np.round(angles, 6))
    print(f"AC PF residual: {residual:.6e}")

    if hasattr(problem, "mu_reference"):
        ref_v, ref_a = problem.voltage_observables(
            problem.mu_reference,
            problem.omega_reference,
        )
        print(
            f"|V| max abs error (p.u.): "
            f"{np.max(np.abs(voltage - ref_v)):.6e}"
        )
        print(
            f"theta max abs error (deg): "
            f"{np.max(np.abs(problem.angle_error_deg(angles, ref_a))):.6e}"
        )

    print(
        f"Residual <= {tolerance:g}: "
        f"{residual <= tolerance}"
    )

    return residual


def solve_one_shot(
    problem,
    bits,
    num_reads,
    lower_lim,
    upper_lim,
    seed=0,
):
    """Solve the original higher-order binary power-flow formulation."""

    problem.generate_discretization(
        n_qubits_per_var=bits,
        binary_representation="adaptive_range",
        lower_lim=lower_lim,
        upper_lim=upper_lim,
    )

    problem.generate_problem_formulation()

    # OpenJij samples the higher-order polynomial directly.
    solver = AnnealingSolverOpenJij()
    solver.setup_solver()

    solver.solve_problem(
        problem,
        num_reads=num_reads,
        seed=seed,
    )

    solutions = problem.analyze_results(result_max=1)

    if not solutions:
        raise RuntimeError(
            "The one-shot solver did not return any samples."
        )

    best = min(solutions, key=lambda s: s["residual"])

    report(
        problem,
        best["mu"],
        best["omega"],
        "One-shot OpenJij result",
        tolerance=1e-4,
    )

    return best["mu"], best["omega"]


def run(
    bus_case,
    bits,
    num_reads,
    output_path,
    lower_lim=-0.8,
    upper_lim=1.2,
    seed=0,
):
    """Load the selected bus case and solve its power-flow problem."""

    case_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"{bus_case}-bus"
    )

    if not case_path.is_dir():
        raise FileNotFoundError(
            f"No {bus_case}-bus case data at {case_path}"
        )

    data = PowerFlowData(str(case_path))
    problem = PowerFlow(data, output_path=output_path)

    mu_ref, omega_ref = problem.compute_reference_solution()
    ref_v, ref_angle = problem.voltage_observables(
        mu_ref,
        omega_ref,
    )

    print("|V|_ref (p.u.):", np.round(ref_v, 4))
    print("theta_ref (deg):", np.round(ref_angle, 3))

    return solve_one_shot(
        problem,
        bits=bits,
        num_reads=num_reads,
        lower_lim=lower_lim,
        upper_lim=upper_lim,
        seed=seed,
    )


def main():
    """Command-line interface for one-shot power flow."""

    parser = argparse.ArgumentParser(
        description="One-shot AC power flow using OpenJij"
    )

    parser.add_argument(
        "--bus",
        type=int,
        default=4,
        choices=[4, 9, 14],
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=6,
        help="Bits per voltage component.",
    )

    parser.add_argument(
        "--num-reads",
        type=int,
        default=200,
        help="Number of OpenJij reads.",
    )

    parser.add_argument(
        "--lower-lim",
        type=float,
        default=-0.8,
    )

    parser.add_argument(
        "--upper-lim",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.bits < 1 or args.num_reads < 1:
        parser.error("bits and num-reads must be positive")

    if args.lower_lim >= args.upper_lim:
        parser.error(
            "lower-lim must be less than upper-lim"
        )

    output_root = (
        Path(__file__).resolve().parent
        / "results"
        / f"powerflow_{args.bus}bus_openjij_one_shot"
    )

    output_path = (
        output_root
        / datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    )

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {output_path}")

    run(
        bus_case=args.bus,
        bits=args.bits,
        num_reads=args.num_reads,
        output_path=str(output_path),
        lower_lim=args.lower_lim,
        upper_lim=args.upper_lim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
