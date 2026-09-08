from collections import defaultdict
import time
from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np

from .solver_base import QAOAParameterOptimizer, QAOASolverBase


class QAOASolverPennylane(QAOASolverBase):
    def setup_device(self, device):
        wires = range(self.n_qubits)

        if device == "MQSSPennylaneDevice":
            if not self.token:
                raise ValueError("A token file must be provided when using MQSSPennylaneDevice.")
            try:
                from mqss.pennylane_adapter.device import MQSSPennylaneDevice
            except ImportError as error:
                raise ImportError(
                    "Optional MQSS dependencies are required to use MQSSPennylaneDevice. "
                    "Install them with `pip install engioptiqa[mqss]`."
                ) from error
            self.dev = MQSSPennylaneDevice(wires=wires, token=self.token, backends="EQE1")
        else:
            self.dev = qp.device(device, wires=wires)

    def qaoa_probability_circuit(self):
        @qp.qnode(self.dev)
        def probability_circuit(betas, gammas):
            self.ansatz(betas, gammas)
            return qp.probs(wires=range(self.n_qubits))

        return probability_circuit

    def qaoa_expectation_circuit(self):
        @qp.qnode(self.dev, interface="auto", diff_method="best")
        def expectation_circuit(betas, gammas):
            self.ansatz(betas, gammas)
            return qp.expval(self.H_cost)

        return expectation_circuit

    def sample_counts(self, betas, gammas, shots):
        @qp.qnode(self.dev)
        def sample_circuit(circuit_betas, circuit_gammas):
            self.ansatz(circuit_betas, circuit_gammas)
            return qp.sample(wires=range(self.n_qubits))

        samples = qp.set_shots(shots)(sample_circuit)(betas, gammas)
        counts = defaultdict(int)
        for row in samples:
            counts[tuple(int(bit) for bit in row)] += 1
        return counts

    def objective_function(self, betas, gammas):
        return self.qaoa_expectation_circuit()(betas, gammas)

    def select_parameters(self, mode, optimization_iterations, tau=None):
        if mode == "fixed":
            return self.fixed_parameters(tau=tau)
        elif mode in {"linear_ramp", "optimize"}:
            return QAOAParameterOptimizer(
                self.objective_function,
                self.num_layers,
                optimization_iterations=optimization_iterations,
            ).optimize(mode)
        else:
            raise ValueError("mode must be 'fixed', 'linear_ramp', or 'optimize'.")

    def execute(self, problem, betas, gammas, circuit, shots):
        if circuit == "probs":
            probs = qp.set_shots(shots)(self.qaoa_probability_circuit())(betas, gammas)
            probs = probs.reshape(-1)
            bitdict_prob_pairs = [
                ({index: int(bit) for index, bit in enumerate(format(value, f"0{self.n_qubits}b"))}, probability)
                for value, probability in enumerate(probs)
            ]
            bitdict_prob_pairs.sort(key=lambda pair: pair[1], reverse=True)
            if shots is not None:
                problem.results = [
                    SimpleNamespace(values=values, energy=0, frequency=int(round(probability * shots)))
                    for values, probability in bitdict_prob_pairs
                    if probability
                ]
            else:
                problem.results = [
                    SimpleNamespace(values=values, energy=0, frequency=1)
                    for values, _ in bitdict_prob_pairs
                ]
            return [probability for _, probability in bitdict_prob_pairs]

        start_time = time.perf_counter()
        counts = self.sample_counts(
            betas, gammas, shots
        )
        print(f"Sampling completed in {time.perf_counter() - start_time:.3f} s")
        return self.store_sample_results(problem, counts, shots)

    def solve_problem(self, problem, num_layers=1, mode="fixed", device="lightning.qubit",
                      circuit="probs", shots=None, tau=None, optimization_iterations=10):
        if num_layers <= 0:
             raise ValueError("Number of layers must be a positive integer.")
        if circuit not in {"probs", "sample"}:
            raise ValueError(f"Unsupported circuit type: {circuit}")
        if circuit == "sample" and shots is None:
            raise ValueError("Number of shots must be specified for sampling mode.")
        if shots is not None and (not isinstance(shots, int) or shots <= 0):
            raise ValueError("Number of shots must be a positive integer.")
        if not isinstance(optimization_iterations, int) or optimization_iterations <= 0:
            raise ValueError("optimization_iterations must be a positive integer.")

        self.prepare_problem_and_ansatz(problem, num_layers)
        self.setup_device(device)
        betas, gammas = self.select_parameters(mode, optimization_iterations, tau=tau)
        return self.execute(
            problem, betas, gammas, circuit, shots
        )
