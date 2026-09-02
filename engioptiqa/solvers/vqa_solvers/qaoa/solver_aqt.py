from collections import defaultdict
import time

import pennylane as qp

from .solver_base import QAOASolverBase


class QAOASolverAQT(QAOASolverBase):
    """Shot-based QAOA solver on AQT backends through OpenQASM and Qiskit."""

    def __init__(self, token_file=None, proxy=None, backend_noise=False, backend=None,
                 optimization_level=3, *args, **kwargs):
        super().__init__(token_file=token_file, proxy=proxy, *args, **kwargs)
        self.backend_noise = backend_noise
        self.backend = backend
        if optimization_level not in range(4):
            raise ValueError("optimization_level must be an integer from 0 to 3.")
        self.optimization_level = optimization_level

    def setup_backend(self):
        if self.backend is not None:
            return

        try:
            from qiskit_aqt_provider import AQTProvider
        except ImportError as error:
            raise ImportError(
                "Optional AQT dependencies are required to use QAOASolverAQT. "
                "Install them with `pip install engioptiqa[aqt]`."
            ) from error

        provider = AQTProvider(self.token or "")
        backend_name = (
            "offline_simulator_noise"
            if self.backend_noise else "offline_simulator_no_noise"
        )
        print(f"Using AQT backend: {backend_name}")
        self.backend = provider.get_backend(backend_name, workspace="default")

    def export_ansatz_qasm(self, betas, gammas):
        qasm_device = qp.device("default.qubit", wires=self.n_qubits)

        @qp.qnode(qasm_device)
        def circuit(circuit_betas, circuit_gammas):
            self.ansatz(circuit_betas, circuit_gammas)
            qp.sample(wires=range(self.n_qubits))

        decomposed_circuit = qp.decompose(
            circuit, gate_set={"H", "RX", "RY", "RZ", "CNOT"}
        )
        return qp.to_openqasm(decomposed_circuit, measure_all=False)(betas, gammas)

    def sample_counts(self, betas, gammas, shots):
        from qiskit import QuantumCircuit
        from qiskit_aqt_provider.primitives import AQTSampler

        circuit = QuantumCircuit.from_qasm_str(self.export_ansatz_qasm(betas, gammas))
        print(f"Running circuit on AQT backend with {shots} shots...")
        sampler = AQTSampler(self.backend)
        sampler.set_transpile_options(optimization_level=self.optimization_level)
        quasi_dist = sampler.run(
            circuits=[circuit], shots=shots
        ).result().quasi_dists[0]
        return self._quasi_dist_to_counts(quasi_dist, shots)

    def _quasi_dist_to_counts(self, quasi_dist, shots):
        if any(probability < 0 for probability in quasi_dist.values()):
            raise ValueError("AQTSampler returned negative quasi-probabilities, which cannot be frequencies.")

        probabilities = list(quasi_dist.items())
        counts = {state: int(probability * shots) for state, probability in probabilities}
        remaining_shots = shots - sum(counts.values())
        for state, _ in sorted(
            probabilities,
            key=lambda item: item[1] * shots - counts[item[0]],
            reverse=True,
        )[:remaining_shots]:
            counts[state] += 1

        return defaultdict(
            int,
            {
                tuple(int(bit) for bit in reversed(format(state, f"0{self.n_qubits}b"))): count
                for state, count in counts.items()
                if count
            },
        )

    def select_parameters(self):
        return self.fixed_parameters()

    def execute(self, problem, betas, gammas, shots):
        start_time = time.perf_counter()
        counts = self.sample_counts(betas, gammas, shots)
        print(f"Sampling completed in {time.perf_counter() - start_time:.3f} s")
        return self.store_sample_results(problem, counts, shots)

    def solve_problem(self, problem, num_layers=1, shots=None):
        if shots is None or not isinstance(shots, int) or shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        self.prepare_problem_and_ansatz(problem, num_layers)
        self.setup_backend()
        betas, gammas = self.select_parameters()
        return self.execute(problem, betas, gammas, shots)
