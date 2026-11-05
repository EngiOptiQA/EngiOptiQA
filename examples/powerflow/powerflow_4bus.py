"""4-bus AC power-flow example.

The script mirrors the workflow used by the structural-analysis examples:

    1. load the case data,
    2. compute a continuous reference solution (the "analytical" baseline),
    3. discretise the rectangular voltage components,
    4. build the Amplify polynomial,
    5. transform to a D-Wave compatible QUBO and solve with simulated
       annealing,
    6. decode the best sample and compare against the reference.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Allow the example to be run directly from the repository root without
# installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engioptiqa import AnnealingSolverDWave, PowerFlow, PowerFlowData


def run(bus_case, n_qubits_per_var, num_reads, output_path):
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'engioptiqa' / 'problems' / 'powerflow' \
        / 'data' / f'{bus_case}-bus'
    if not data_path.exists():
        raise FileNotFoundError(f'No data folder for {bus_case}-bus at {data_path}')

    # Case data
    # =========
    data = PowerFlowData(str(data_path))
    problem = PowerFlow(data, output_path=output_path)

    # Continuous reference (Newton-class behaviour via L-BFGS-B on the
    # exact same residual the QUBO minimises).
    # ===================================================================
    mu_ref, omega_ref = problem.compute_reference_solution()
    print('|V|_ref :', np.round(np.sqrt(mu_ref ** 2 + omega_ref ** 2), 4))
    print('theta_ref (deg):', np.round(np.degrees(np.arctan2(omega_ref, mu_ref)), 3))

    # Discretisation of mu/omega through RealNumber adaptive_range
    # =============================================================
    problem.generate_discretization(
        n_qubits_per_var=n_qubits_per_var,
        binary_representation='adaptive_range',
        lower_lim=-1.5,
        upper_lim=1.5,
    )

    # Polynomial objective via Amplify
    # =================================
    problem.generate_problem_formulation()

    # QUBO conversion and simulated annealing
    # =======================================
    problem.transform_to_dwave()

    solver = AnnealingSolverDWave()
    solver.setup_solver(solver_type='simulated_annealing')
    solver.solve_problem(problem, num_reads=num_reads, seed=0)

    # Analysis and comparison against the reference
    # =============================================
    problem.analyze_results(result_max=1)


def main():
    parser = argparse.ArgumentParser(
        description='AC power-flow example based on EngiOptiQA.'
    )
    parser.add_argument('--bus', type=int, default=4,
                        choices=[4, 9, 14],
                        help='Bus case to load (default: 4).')
    parser.add_argument('--bits', type=int, default=6,
                        help='Bits per voltage component (default: 6).')
    parser.add_argument('--num-reads', type=int, default=200,
                        help='Number of SA reads (default: 200).')
    args = parser.parse_args()

    output_root = Path(__file__).resolve().parent / 'results' / f'powerflow_{args.bus}bus_sa'
    output_path = output_root / datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Output folder: {output_path}')

    run(args.bus, args.bits, args.num_reads, str(output_path))


if __name__ == '__main__':
    main()
