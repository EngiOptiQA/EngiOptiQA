from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import math
from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np
from pennylane import qaoa


class QAOASolverBase(ABC):
    """Shared QAOA formulation and workflow contract for backend-specific solvers."""

    def __init__(self, token_file=None, proxy=None, *args, **kwargs):
        self.proxy = proxy
        self.token = None
        if token_file is not None:
            with open(token_file, "r", encoding="utf-8") as token_handle:
                self.token = token_handle.read().strip()

    def prepare_problem_and_ansatz(self, problem, num_layers):
        binary_poly_dict = problem.binary_model.objective.as_dict()
        ising_poly_dict = self.convert_binary_to_ising(binary_poly_dict)
        self.n_qubits = max(
            (max(term) for term in ising_poly_dict if term), default=-1
        ) + 1
        self.construct_cost_hamiltonian(ising_poly_dict, normalize=True)
        self.construct_mixer_hamiltonian(scaling=True)
        self.num_layers = num_layers
        self.ansatz = self.qaoa_ansatz

    def convert_binary_to_ising(self, binary_poly_dict):
        ising_poly_dict = defaultdict(float)
        for term, binary_coeff in binary_poly_dict.items():
            degree = len(term)
            for subset_size in range(degree + 1):
                factor = (-1) ** subset_size / 2 ** degree if degree else 1.0
                for subset in itertools.combinations(term, subset_size):
                    ising_poly_dict[tuple(sorted(subset))] += binary_coeff * factor
        return ising_poly_dict

    def construct_cost_hamiltonian(self, ising_poly_dict, normalize=True):
        ising_poly_dict.pop((), 0.0)
        coeffs = []
        operators = []
        for qubits, coefficient in ising_poly_dict.items():
            operator = qp.PauliZ(qubits[0])
            for qubit in qubits[1:]:
                operator = operator @ qp.PauliZ(qubit)
            coeffs.append(coefficient)
            operators.append(operator)

        self.H_cost = qp.Hamiltonian(coeffs, operators)
if normalize and self.H_cost.coeffs:
    coeffs = np.array(self.H_cost.coeffs, dtype=float)
    max_coefficient = np.max(np.abs(coeffs))
    if max_coefficient:
        self.H_cost = qp.Hamiltonian((coeffs / max_coefficient).tolist(), self.H_cost.ops)

    def construct_mixer_hamiltonian(self, scaling=True):
        scale_factor = 1.0
if scaling and self.H_cost.coeffs:
    mean_coefficient = float(np.mean(np.abs(self.H_cost.coeffs)))
    scale_factor = mean_coefficient if mean_coefficient and math.isfinite(mean_coefficient) else 1.0
        self.H_mixer = qp.Hamiltonian(
            [scale_factor] * self.n_qubits,
            [qp.PauliX(qubit) for qubit in range(self.n_qubits)],
        )

    def qaoa_layer(self, beta, gamma):
        qaoa.cost_layer(gamma, self.H_cost)
        qaoa.mixer_layer(-beta, self.H_mixer)

    def qaoa_ansatz(self, betas, gammas):
        for wire in range(self.n_qubits):
            qp.Hadamard(wires=wire)
        qp.layer(self.qaoa_layer, self.num_layers, betas, gammas)

    def fixed_parameters(self):
        return (
            np.linspace(1, 0, self.num_layers),
            np.linspace(0, 1, self.num_layers),
        )

    def store_sample_results(self, problem, counts, shots):
        results = []
        for bit_tuple, count in counts.items():
            values = {index: int(bit) for index, bit in enumerate(bit_tuple)}
            results.append((values, count / shots, count))
        results.sort(key=lambda result: result[1], reverse=True)
        problem.results = [
            SimpleNamespace(values=values, energy=0, frequency=count)
            for values, _, count in results
        ]
        return [probability for _, probability, _ in results]

    @abstractmethod
    def select_parameters(self, **kwargs):
        """Return QAOA beta and gamma parameters."""

    @abstractmethod
    def execute(self, problem, betas, gammas, **kwargs):
        """Execute the configured QAOA circuit and store results."""

    @abstractmethod
    def solve_problem(self, problem, num_layers=1, **kwargs):
        """Solve an optimization problem using a backend-specific QAOA evaluator."""

class QAOAParameterOptimizer:
    """Optimize QAOA parameters against an evaluator-provided objective."""

    def __init__(self, objective_function, num_layers, optimization_iterations=10,
                 optimizer=None):
        self.objective_function = objective_function
        self.num_layers = num_layers
        self.optimization_iterations = optimization_iterations
        self.status_interval = math.ceil(optimization_iterations / 10)
        self.optimizer = optimizer or qp.AdamOptimizer()

    def optimize(self, mode):
        if mode == "linear_ramp":
            return self._optimize_linear_ramp()
        if mode == "optimize":
            return self._optimize_individual_parameters()
        raise ValueError("mode must be 'fixed', 'linear_ramp', or 'optimize'.")

    def _optimize_individual_parameters(self):
        initial_betas = np.random.uniform(0, 1, self.num_layers)
        initial_gammas = np.random.uniform(0, 1, self.num_layers)
        params = np.concatenate([initial_betas, initial_gammas], requires_grad=True)

        def objective(values):
            return self.objective_function(
                values[:self.num_layers], values[self.num_layers:]
            )

        for iteration in range(self.optimization_iterations):
            params = self.optimizer.step(objective, params)
            self._print_status(iteration, objective(params))
        return params[:self.num_layers], params[self.num_layers:]

    def _optimize_linear_ramp(self):
        params = np.array([1.0, 1.0], requires_grad=True)

        def objective(values):
            return self.objective_function(
                np.linspace(1, 0, self.num_layers) * values[0],
                np.linspace(0, 1, self.num_layers) * values[1],
            )

        for iteration in range(self.optimization_iterations):
            params = self.optimizer.step(objective, params)
            self._print_status(iteration, objective(params))
        return (
            np.linspace(1, 0, self.num_layers) * params[0],
            np.linspace(0, 1, self.num_layers) * params[1],
        )

    def _print_status(self, iteration, objective_value):
        completed_iterations = iteration + 1
        if (
            completed_iterations % self.status_interval == 0
            or completed_iterations == self.optimization_iterations
        ):
            print(
                f"Optimization iteration {completed_iterations}/"
                f"{self.optimization_iterations}: {objective_value:.4f}"
            )
