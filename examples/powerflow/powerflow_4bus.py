"""EngiOptiQA power-flow example: one-shot or iterative AQPF.

One-shot preserves the original absolute RealNumber representation. Iterative
AQPF rebuilds the same P/Q mismatch polynomial around a moving voltage center.
The solver remains EngiOptiQA's AnnealingSolverDWave in both modes.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engioptiqa import AnnealingSolverDWave, PowerFlow, PowerFlowData


def aqpf_delta_mu(it):
    """Use the schedule in the supplied AQPF(v0.1.0).py script."""
    base = 0.04 * (100 - it) / 100
    if it < 4:
        return max(-0.05 * it + base, 0.0005)
    if it < 40:
        return max(-0.001 * it + base, 0.0005)
    return 0.0005


def aqpf_delta_omega(it):
    base = 0.02 * (100 - it) / 100
    if it < 4:
        return max(-0.05 * it + base, 0.0005)
    if it < 40:
        return max(-0.0005 * it + base, 0.0005)
    return 0.0005


def report(problem, mu, omega, title, tolerance):
    voltage, angles = problem.voltage_observables(mu, omega)
    residual = problem.compute_residual(mu, omega)
    print(f'\n{title}')
    print('mu:', np.round(mu, 6))
    print('omega:', np.round(omega, 6))
    print('|V| (p.u.):', np.round(voltage, 6))
    print('theta (deg):', np.round(angles, 6))
    print(f'AC PF residual: {residual:.6e}')
    if hasattr(problem, 'mu_reference'):
        ref_v, ref_a = problem.voltage_observables(
            problem.mu_reference, problem.omega_reference)
        print(f'|V| max abs error (p.u.): {np.max(np.abs(voltage-ref_v)):.6e}')
        print(f'theta max abs error (deg): '
              f'{np.max(np.abs(problem.angle_error_deg(angles, ref_a))):.6e}')
    print(f'Converged (residual <= {tolerance:g}): {residual <= tolerance}')
    return residual


def solve_one_shot(problem, bits, num_reads, lower_lim, upper_lim):
    """Original workflow with configurable absolute voltage bounds."""
    problem.generate_discretization(
        n_qubits_per_var=bits,
        binary_representation='adaptive_range',
        lower_lim=lower_lim,
        upper_lim=upper_lim,
    )
    problem.generate_problem_formulation()
    problem.transform_to_dwave()
    solver = AnnealingSolverDWave()
    solver.setup_solver(solver_type='simulated_annealing')
    solver.solve_problem(problem, num_reads=num_reads, seed=0)
    solutions = problem.analyze_results(result_max=1)
    if not solutions:
        raise RuntimeError('The one-shot solver did not return any samples.')
    best = min(solutions, key=lambda s: s['residual'])
    report(problem, best['mu'], best['omega'], 'One-shot result', tolerance=1e-4)
    return best['mu'], best['omega']


def solve_iterative(problem, num_reads, max_iter, tolerance, seed):
    """AQPF's local signed-increment search with monotonic mismatch acceptance.

    Unlike the provided AQPF script's random-neighbor postprocessing, this
    implementation evaluates every *returned* annealer sample and retains the
    best actual AC residual. The original current center is always retained if
    no sample improves on it. This makes residual evolution nonincreasing but
    does not guarantee convergence or reproduce the AQPF paper's exact trace.
    """
    data = problem.data
    mu, omega = np.ones(data.n_bus), np.zeros(data.n_bus)
    residual = problem.compute_residual(mu, omega)
    print(f'Initial AQPF residual: {residual:.6e}')
    solver = AnnealingSolverDWave()
    solver.setup_solver(solver_type='simulated_annealing')

    for it in range(max_iter):
        if residual <= tolerance:
            break
        delta_mu, delta_omega = aqpf_delta_mu(it), aqpf_delta_omega(it)
        problem.generate_aqpf_discretization(mu, omega, delta_mu, delta_omega)
        problem.generate_problem_formulation()
        problem.transform_to_dwave()
        solver.solve_problem(problem, num_reads=num_reads, seed=seed + it)

        candidate_mu, candidate_omega, candidate_res = mu.copy(), omega.copy(), residual
        for result in problem.results:
            sample_mu, sample_omega = problem.decode_voltages(result)
            sample_mu[data.ref_bus] = 1.0
            sample_omega[data.ref_bus] = 0.0
            score = problem.compute_residual(sample_mu, sample_omega)
            if np.isfinite(score) and score < candidate_res:
                candidate_mu = sample_mu.copy()
                candidate_omega = sample_omega.copy()
                candidate_res = score

        mu, omega, residual = candidate_mu, candidate_omega, candidate_res
        print(f'Iteration {it + 1:3d}: residual={residual:.6e}, '
              f'delta_mu={delta_mu:.6g}, delta_omega={delta_omega:.6g}')

    report(problem, mu, omega, 'Iterative AQPF result', tolerance)
    if residual > tolerance:
        print('WARNING: iterative AQPF did not meet the requested tolerance.')
    return mu, omega


def run(bus_case, bits, num_reads, output_path, mode='one-shot',
        max_iter=100, tolerance=1e-4, lower_lim=-1.5, upper_lim=1.5, seed=0):
    case_path = Path(__file__).resolve().parent / 'data' / f'{bus_case}-bus'
    # Also support the original branch before PR #19 data relocation.
    if not case_path.exists():
        fallback = (Path(__file__).resolve().parents[2]
                    / 'engioptiqa' / 'problems' / 'powerflow'
                    / 'data' / f'{bus_case}-bus')
        if fallback.exists():
            case_path = fallback
    if not case_path.exists():
        raise FileNotFoundError(f'No {bus_case}-bus case data at {case_path}')
    data = PowerFlowData(str(case_path))
    problem = PowerFlow(data, output_path=output_path)
    mu_ref, omega_ref = problem.compute_reference_solution()
    ref_v, ref_angle = problem.voltage_observables(mu_ref, omega_ref)
    print('|V|_ref (p.u.):', np.round(ref_v, 4))
    print('theta_ref (deg):', np.round(ref_angle, 3))
    if mode == 'one-shot':
        return solve_one_shot(problem, bits, num_reads, lower_lim, upper_lim)
    if mode == 'iterative':
        return solve_iterative(problem, num_reads, max_iter, tolerance, seed)
    raise ValueError(f'Unknown mode: {mode}')


def main():
    parser = argparse.ArgumentParser(description='AC power flow with EngiOptiQA')
    parser.add_argument('--bus', type=int, default=4, choices=[4, 9, 14])
    parser.add_argument('--mode', choices=['one-shot', 'iterative'], default='one-shot')
    parser.add_argument('--bits', type=int, default=6,
                        help='Bits per component in one-shot mode only.')
    parser.add_argument('--num-reads', type=int, default=200)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--lower-lim', type=float, default=-0.8,
                        help='One-shot absolute rectangular-voltage lower bound.')
    parser.add_argument('--upper-lim', type=float, default=1.2,
                        help='One-shot absolute rectangular-voltage upper bound.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.num_reads < 1 or args.bits < 1 or args.max_iter < 1 or args.tolerance <= 0:
        parser.error('bits, num-reads, max-iter and tolerance must be positive')
    if args.lower_lim >= args.upper_lim:
        parser.error('lower-lim must be less than upper-lim')

    output_root = (Path(__file__).resolve().parent / 'results'
                   / f'powerflow_{args.bus}bus_sa_{args.mode}')
    output_path = output_root / datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Output folder: {output_path}')
    run(args.bus, args.bits, args.num_reads, str(output_path),
        mode=args.mode, max_iter=args.max_iter, tolerance=args.tolerance,
        lower_lim=args.lower_lim, upper_lim=args.upper_lim, seed=args.seed)


if __name__ == '__main__':
    main()
