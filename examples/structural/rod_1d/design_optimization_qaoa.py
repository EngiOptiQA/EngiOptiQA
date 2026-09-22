import sys
from datetime import datetime
from pathlib import Path

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from engioptiqa import (
    QAOASolverPennylane,
    DesignOptimizationProblemRod1D,
    Rod1D,
)

# Get the directory containing this script
script_directory = Path(__file__).resolve().parent

# Create an output folder with a timestamp
results_root = script_directory / "results" / "design_optimization_qaoa"
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

optimization_problem = DesignOptimizationProblemRod1D(rod_1d, g, A_choice=A_choices, output_path=output_path)

# Analytical Solution
# ===================
optimization_problem.compute_analytical_solution()

# Numerical Solution
# ==================

# QAOA Solver based on PennyLane
# -------------------------------
qaoa_solver = QAOASolverPennylane()
p = 5
shots = 500

# Discretization through Binary Representation of Real-Valued Nodal Coefficients and Cross Section Choice
# -------------------------------------------------------------------------------------------------------
binary_representation = 'normalized'
n_qubits_per_var = 3

optimization_problem.generate_discretization(n_qubits_per_var, binary_representation)

# Problem Formulation Using the Amplify SDK
# --------------------------------------
penalty_weight = 7.5e2
optimization_problem.generate_problem_formulation(penalty_weight=penalty_weight)

probs = qaoa_solver.solve_problem(
                optimization_problem,
                num_layers=p,
                mode="fixed",
                device="lightning.qubit",
                circuit="sample",
                shots=shots,
            )

# Get the Best Solution, i.e., with Minimum Objective Value
# =========================================================
best_solution = optimization_problem.get_best_solution()

# Plot Force Distribution for the Best Solution
# ---------------------------------------------
optimization_problem.plot_force(
    optimization_problem.force_analytic,
    best_solution['force'],
    subtitle='QAOA',
    file_name= str(output_path / "force_qaoa"),
    save_fig = True,
    save_tikz = True
)