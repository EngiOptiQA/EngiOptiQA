import sys
from datetime import datetime
import numpy as np
from pathlib import Path

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engioptiqa import AnnealingSolverDWave, TrussStructure

# Get the directory containing this script
script_directory = Path(__file__).resolve().parent

# Create an output folder with a timestamp
results_root = script_directory / "results" / "analysis_sa"
output_path = results_root / datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
output_path.mkdir(parents=True, exist_ok=True)
print(f"Created output folder: {output_path}")

def create_truss_structure_3_elements(ts, A, E, load, visualize=True, subtitle=''):
    ts.add_node(0, (0,0))  # Node 0 at (0,0)
    ts.add_node(1, (1,0))  # Node 1 at (1,0)
    ts.add_node(2, (0,1))  # Node 2 at (0,1)

    ts.add_member(0, 1, A=A, E=E)  # Member 0 from Node 0 to Node 1
    ts.add_member(1, 2, A=A, E=E)  # Member 1 from Node 1 to Node 2

    ts.add_load(1, load)  # Vertical load of -100 N at Node 1

    ts.add_support(0, True, True)  # Fixed support at Node 0
    ts.add_support(2, True, True)  # Fixed support at Node 2

    if visualize:
        ts.visualize(subtitle)


# The Analysis Problem
# ====================
# Define the truss structure with the following cross-sectional area A, Young's modulus E,
# and vertical load of -100 kN.
A = 0.5; E = 2e11; load = (0, -100e3)

ts = TrussStructure(output_path=output_path)
create_truss_structure_3_elements(ts, A, E, load, visualize=False, subtitle='Reference')

# Reference Solution
# ==================
ts_ref = TrussStructure()
create_truss_structure_3_elements(ts_ref, A, E, load, visualize=False, subtitle='Reference')
ts.set_reference_solution(ts_ref)

# Numerical Solution
# ==================

# Discretization through Binary Representation of Real-Valued Member Stress
# -------------------------------------------------------------------------
binary_representation = 'range'
n_qubits_per_var = 10
ts.generate_discretization(n_qubits_per_var=n_qubits_per_var,
                           binary_representation=binary_representation,
                           lower_lim=-2.5e5, upper_lim=3e5)

# Problem Formulation Using the Amplify SDK
# -----------------------------------------
penalty_weight = 1e2
ts.generate_problem_formulation(penalty_weight=penalty_weight)
coeff_dict = ts.complementary_energy_poly.as_dict()

# Transform Amplify Problem for D-Wave Solver
# -------------------------------------------
ts.transform_to_dwave()

# Simulated Annealing Solver from D-Wave
# --------------------------------------
annealing_solver_sa = AnnealingSolverDWave()
annealing_solver_sa.setup_solver(solver_type='simulated_annealing')

# Solve QUBO Problem by Simulated Annealing
# -----------------------------------------
annealing_solver_sa.solve_problem(ts, num_reads=50)

# Analyze Solution
# ================
solutions_sa = ts.analyze_results(result_max=0)
objectives = [s["objective"] for s in solutions_sa]
i_best = int(np.argmin(objectives))
best_solution = solutions_sa[i_best]

rel_error_forces, _, rel_error_compliance = ts.compare_with_reference_solution(best_solution)

print('Compliance:')
print('===========')
print(f'  Rel. Diff: {rel_error_compliance:.2e}')
print('Force:')
print('======')
for i_member, member in enumerate(ts.members):
    print(f'  Member {i_member}')
    print(f'    Rel. Diff: {rel_error_forces[i_member]:.2e}')