import sys
from datetime import datetime
import numpy as np
from pathlib import Path

# Make sure the repo root is on the path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from engioptiqa import AnnealingSolverDWave, DesignOptimizationProblem, Rod1D

# Get the directory containing this script
script_directory = Path(__file__).resolve().parent

# Create an output folder with a timestamp
results_root = script_directory / "results" / "design_optimization_sa"
output_path = results_root / datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
output_path.mkdir(parents=True, exist_ok=True)
print(f"Created output folder: {output_path}")

# The Design Optimization Problem
# ===============================
# Define the design optimization problem for the one-dimensional rod under self-weight loading
# through body force density g.
g = 1.5
# Rod with n_comp components and of length L.
n_comp = 2; L = 1.5; A_choices = [0.25, 0.5]; rod_1d = Rod1D(n_comp, L)

optimization_problem = DesignOptimizationProblem(rod_1d, g, A_choice=A_choices, output_path=output_path)

# Analytical Solution
# ===================
optimization_problem.compute_analytical_solution()

# Numerical Solution
# ==================

# Simulated Annealing Solver from D-Wave
# --------------------------------------
annealing_solver_sa = AnnealingSolverDWave()
annealing_solver_sa.setup_solver(solver_type='simulated_annealing')

# Discretization through Binary Representation of Real-Valued Nodal Coefficients and Cross Section Choice
# -------------------------------------------------------------------------------------------------------
binary_representation = 'normalized'
n_qubits_per_node = 3

optimization_problem.generate_discretization(n_qubits_per_node, binary_representation)

# QUBO Formulation Using the Amplify SDK
# --------------------------------------
penalty_weight = 7.5e2
optimization_problem.generate_qubo_formulation(penalty_weight=penalty_weight)

# Transform Amplify Problem for D-Wave Solver
# -------------------------------------------
optimization_problem.transform_to_dwave()

# Solve QUBO Problem by Simulated Annealing
# -----------------------------------------
annealing_solver_sa.solve_qubo_problem(
    optimization_problem,
    num_reads=200,
    )

# Analyze Solution
# ================
solutions_sa = optimization_problem.analyze_results(result_max=0)

# Get the Best Solution, i.e., with Minimum Objective Value
# ---------------------------------------------------------
objectives = [d['objective'] for d in solutions_sa]
i_min = np.argsort(objectives)
i_sol = i_min[0]
solution = solutions_sa[i_sol]

# Plot Force Distribution for the Best Solution
# ---------------------------------------------
optimization_problem.plot_force(
    optimization_problem.force_analytic,
    solution['force'],
    subtitle='Simulated Annealing',
    file_name= str(output_path / "force_sa"),
    save_fig = True,
    save_tikz = True
)



