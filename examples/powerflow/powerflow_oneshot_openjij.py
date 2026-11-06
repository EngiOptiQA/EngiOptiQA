"""One-shot EngiOptiQA PUBO solved with OpenJij.

Examples:
    python examples/powerflow/powerflow_oneshot_openjij.py --bus 2 --bits 2
    python examples/powerflow/powerflow_oneshot_openjij.py --bus 2 --bits 3 --encoding increment_decrement

Use the same --encoding/--bits/--bounds or --delta settings in the QAOA example to
compare the exact same discrete objective.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from engioptiqa import AnnealingSolverOpenJij, PowerFlow, PowerFlowData

def print_voltage_report(problem, mu, omega, tolerance=1e-4):
    magnitude, angle = problem.voltage_observables(mu, omega)
    residual = float(problem.compute_residual(mu, omega))
    print("\nBus          mu        omega        |V|       angle (deg)")
    for i in range(len(mu)):
        print(f"{i:3d} {mu[i]:12.6f} {omega[i]:12.6f} "
              f"{magnitude[i]:12.6f} {angle[i]:12.6f}")
    print(f"Physical AC residual: {residual:.9e}; "
          f"within tolerance {tolerance:g}: {residual <= tolerance}")
    return residual, magnitude, angle


def verify_model(problem, bits):
    actual = len(problem.binary_model.get_variables())
    expected = 2 * bits * (problem.data.n_bus - 1)
    if actual != expected:
        raise RuntimeError(f"Expected {expected} binary variables, found {actual}")
    return actual




def main():
    parser = argparse.ArgumentParser(description='One-shot AC power flow with EngiOptiQA/OpenJij')
    parser.add_argument('--bus', type=int, choices=[2, 4, 9, 14], default=2)
    parser.add_argument('--bits', type=int, default=2)
    parser.add_argument('--encoding', choices=['adaptive_range_binary', 'range_binary', 'increment_decrement'], default='adaptive_range_binary')
    parser.add_argument('--lower-lim', type=float, default=None,
                        help='Optional legacy common lower bound for both components')
    parser.add_argument('--upper-lim', type=float, default=None,
                        help='Optional legacy common upper bound for both components')
    parser.add_argument('--mu-lower', type=float, default=None)
    parser.add_argument('--mu-upper', type=float, default=None)
    parser.add_argument('--omega-lower', type=float, default=None)
    parser.add_argument('--omega-upper', type=float, default=None)
    parser.add_argument('--mu-base', type=float, default=1.0)
    parser.add_argument('--omega-base', type=float, default=0.0)
    parser.add_argument('--delta-mu', type=float, default=0.04)
    parser.add_argument('--delta-omega', type=float, default=0.06)
    parser.add_argument('--num-reads', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--no-reference', action='store_true')
    args = parser.parse_args()

    if args.encoding == 'increment_decrement' and args.bits != 2:
        parser.error('increment_decrement requires --bits 2')
    if args.bits < 1 or args.num_reads < 1 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('bits, num-reads and tolerance must be positive')

    if (args.lower_lim is None) != (args.upper_lim is None):
        parser.error('--lower-lim and --upper-lim must be provided together')
    if args.lower_lim is not None and args.lower_lim >= args.upper_lim:
        parser.error('Common voltage bounds must be increasing')
    for low, high in ((args.mu_lower, args.mu_upper), (args.omega_lower, args.omega_upper)):
        if low is not None and high is not None and low >= high:
            parser.error('Component voltage bounds must be increasing')
    case = HERE / 'data' / f'{args.bus}-bus'
    if not case.is_dir():
        parser.error(f'Case directory does not exist: {case}')
    output = (HERE / 'results' / f'powerflow_{args.bus}bus_openjij_oneshot' /
              datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    output.mkdir(parents=True, exist_ok=True)

    data = PowerFlowData(str(case))
    if data.n_bus != args.bus:
        raise ValueError(f'Expected {args.bus} buses, got {data.n_bus}')
    problem = PowerFlow(data, output_path=str(output))
    print(f'Case: {args.bus}-bus; solver=OpenJij; encoding={args.encoding}; bits={args.bits}', flush=True)
    if data.pv_buses:
        print('WARNING: The current PowerFlow objective does not enforce PV voltage setpoints.')

    if not args.no_reference:
        try:
            ref_mu, ref_omega = problem.compute_reference_solution()
            print('\n=== Continuous reference ===')
            print_voltage_report(problem, ref_mu, ref_omega, args.tolerance)
        except RuntimeError as exc:
            print(f'WARNING: Reference minimizer did not converge: {exc}')

    if args.encoding == "increment_decrement":
        problem.generate_discretization(
            n_qubits_per_var=args.bits,
            binary_representation=args.encoding,
            mu_center=args.mu_base, omega_center=args.omega_base,
            delta_mu=args.delta_mu, delta_omega=args.delta_omega,
        )
    else:
        problem.generate_discretization(
            n_qubits_per_var=args.bits,
            binary_representation=args.encoding,
            lower_lim=args.lower_lim, upper_lim=args.upper_lim,
            mu_lower=args.mu_lower, mu_upper=args.mu_upper,
            omega_lower=args.omega_lower, omega_upper=args.omega_upper,
        )
    problem.generate_problem_formulation()
    n = verify_model(problem, args.bits)
    print(f'Logical binary variables: {n}', flush=True)
    print(f'Package discretization: {args.encoding}')
    if args.encoding != 'increment_decrement':
        print(f'Real range: {problem.mu_range}; imaginary range: {problem.omega_range}')

    sampler = AnnealingSolverOpenJij()
    sampler.setup_solver()
    sampler.solve_problem(problem, num_reads=args.num_reads, seed=args.seed)

    solutions = problem.analyze_results(result_max=1)
    if not solutions:
        raise RuntimeError('OpenJij returned no samples')
    best = min(solutions, key=lambda x: x['residual'])
    mu, omega = np.asarray(best['mu']), np.asarray(best['omega'])
    _, magnitudes, angles = print_voltage_report(problem, mu, omega, args.tolerance)
    with (output / 'voltages.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bus', 'mu', 'omega', 'magnitude_pu', 'angle_deg'])
        for i in range(data.n_bus):
            writer.writerow([i, mu[i], omega[i], magnitudes[i], angles[i]])
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()