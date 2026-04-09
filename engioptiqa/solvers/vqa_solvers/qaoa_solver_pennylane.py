from collections import defaultdict
import itertools
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
import numpy
import pennylane as qml
from pennylane import numpy as np
from pennylane import qaoa
from types import SimpleNamespace

class QAOASolver:
    def __init__(self, token_file=None, proxy=None, *args, **kwargs):
        self.proxy = proxy
        if token_file is not None:
            self.token = open(token_file,"r").read().replace('\n', '')

class QAOASolverPennylane(QAOASolver):
    def __init__(self, token_file=None, proxy=None, *args, **kwargs):
        super().__init__(token_file=token_file, proxy=proxy, *args, **kwargs)

    def setup_device(self, device='lightning.qubit', shots=None):
        wires = range(self.n_qubits)

        if device == 'MQSSPennylaneDevice':
            self.dev = MQSSPennylaneDevice(wires=wires, token=self.token, backends='EQE1')
        else:
            self.dev =  qml.device(device, wires=wires, shots=shots)


    def convert_binary_to_ising(self, binary_poly_dict):
        ising_poly_dict = defaultdict(float)

        ising_poly_dict = defaultdict(float)
        # Iterate over each term in the binary polynomial
        for term, binary_coeff in binary_poly_dict.items():
            degree = len(term)
            # For each term, we need to consider all subsets of the variables
            # in the term to convert to Ising form
            for r in range(0, degree + 1):
                # The factor accounts for the transformation from binary to Ising variables
                factor = ((-1)**r) / (2**degree) if degree > 0 else 1.0
                for subset in itertools.combinations(term, r):
                    subset = tuple(sorted(subset))
                    ising_poly_dict[subset] += binary_coeff * factor

        return ising_poly_dict

    def construct_cost_hamiltonian(self, ising_poly_dict, normalize=True):

        coeffs = [] # List of coefficients for the Hamiltonian terms
        ops = [] # List of corresponding operators

        # Remove constant term
        ising_poly_dict.pop((), 0.0)

        for qubits, c in ising_poly_dict.items():
            op = qml.PauliZ(qubits[0])
            for q in qubits[1:]:
                op = op @ qml.PauliZ(q)
            coeffs.append(c)
            ops.append(op)

        self.H_cost = qml.Hamiltonian(coeffs, ops)

        if normalize:
            coeffs = np.array(self.H_cost.coeffs, dtype=float)

            coeff_abs_max = np.max(np.abs(coeffs))
            coeffs_norm = (coeffs / coeff_abs_max).tolist()
            self.H_cost = qml.Hamiltonian(coeffs_norm, self.H_cost.ops)

        # print("Cost Hamiltonian:", self.H_cost)

    def construct_mixer_hamiltonian(self, scaling=True):

        scale_factor = 1.0
        if scaling:
            mean_coeff_abs = np.mean(np.abs(self.H_cost.coeffs))
            scale_factor = mean_coeff_abs if mean_coeff_abs != 0 else 1.0
        self.H_mixer = qml.Hamiltonian([scale_factor]*self.n_qubits, [qml.PauliX(i) for i in range(self.n_qubits)])

        # print("Mixer Hamiltonian:", self.H_mixer)

    def qaoa_layer(self, beta, gamma):
        qaoa.cost_layer(gamma, self.H_cost)
        qaoa.mixer_layer(-beta, self.H_mixer)  # Watch out for the minus sign!

    def qaoa_ansatz(self, betas, gammas):
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        qml.layer(self.qaoa_layer, self.num_layers, betas, gammas)

    def qaoa_probability_circuit(self):

        @qml.qnode(self.dev)
        def probability_circuit(betas, gammas):
            self.ansatz(betas, gammas)
            return qml.probs(wires=range(self.n_qubits))

        return probability_circuit

    def qaoa_expectation_circuit(self):

        @qml.qnode(self.dev, interface="auto", diff_method="best")
        def expectation_circuit(betas, gammas):
            self.ansatz(betas, gammas)
            return qml.expval(self.H_cost)

        return expectation_circuit

    def objective_function(self, betas, gammas):
        expectation_circuit = self.qaoa_expectation_circuit()
        return expectation_circuit(betas, gammas)

    def optimize_parameters(self, initial_betas, initial_gammas):

        def objective(params):
            betas = params[:self.num_layers]
            gammas = params[self.num_layers:]
            return self.objective_function(betas, gammas)

        params = np.concatenate([initial_betas, initial_gammas], requires_grad=True)

        for i in range(10):
            params = self.optimizer.step(objective, params)
            print(f"Optimization iteration {i+1}/10: {objective(params):.4f}")

        return params[:self.num_layers], params[self.num_layers:]

    def optimize_linear_ramp_parameters(self, dbeta_initial, dgamma_initial):

        def objective(params):
            dbeta, dgamma = params
            betas = np.linspace(1, 0, self.num_layers)  * dbeta
            gammas = np.linspace(0, 1, self.num_layers) * dgamma
            return self.objective_function(betas, gammas)

        params = np.array([dbeta_initial, dgamma_initial], requires_grad=True)
        for i in range(3):
            params = self.optimizer.step(objective, params)
            print(f"Optimization iteration {i+1}/3: {objective(params):.4f}")

        return params[0], params[1]

    def sort_bitstrings_by_probs(self, probs):
        print(f'sum(probs): {sum(probs)}')
        bitdicts = [
            {i: int(b) for i, b in enumerate(format(x, f"0{self.n_qubits}b"))}
            for x in range(2**self.n_qubits)
        ]
        print("Number of solutions:", len(bitdicts))
        pairs = list(zip(bitdicts, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)

        best_bitstring, best_prob = pairs[0]
        print("Most likely:", best_bitstring, best_prob)

        return pairs

    def solve_problem(self, problem, num_layers=1, mode='fixed', shots=None):
        print("Solving problem with Pennylane QAOA solver")

        # Convert the binary polynomial to an Ising polynomial
        binary_poly_dict = problem.binary_quadratic_model.objective.asdict()
        ising_poly_dict = self.convert_binary_to_ising(binary_poly_dict)

        # Determine the number of qubits
        self.n_qubits = max([max(k) for k in ising_poly_dict.keys() if len(k) > 0], default=-1) + 1

        # Build PennyLane Hamiltonians
        self.construct_cost_hamiltonian(ising_poly_dict, normalize=True)
        self.construct_mixer_hamiltonian(scaling=True)

        # Set up the device
        self.setup_device(shots=shots)

        # Prepare the QAOA ansatz
        self.num_layers = num_layers
        self.ansatz = self.qaoa_ansatz

        # Determine the parameters for the QAOA circuit
        if mode == 'fixed':
            betas = np.linspace(1, 0, self.num_layers)
            gammas = np.linspace(0, 1, self.num_layers)
        else:
            self.optimizer = qml.AdamOptimizer()
            if mode == 'linear_ramp':
                dbeta_initial = 1.
                dgamma_initial = 1.
                dbeta, dgamma = self.optimize_linear_ramp_parameters(dbeta_initial, dgamma_initial)
                betas = np.linspace(1, 0, self.num_layers) * dbeta
                gammas = np.linspace(0, 1, self.num_layers) * dgamma
            else:
                betas_initial = np.random.uniform(0, 1, self.num_layers)
                gammas_initial = np.random.uniform(0, 1, self.num_layers)
                betas, gammas = self.optimize_parameters(betas_initial, gammas_initial)

        # For the final circuit, compute probabilities of all bitstrings
        probability_circuit = self.qaoa_probability_circuit()
        probs = probability_circuit(betas, gammas)
        if probs.ndim == 2 and probs.shape[0] == 1:
            probs = probs[0]
        probs = probs.reshape(-1)

        # Sort bitstrings by probability and store results in the problem
        bitdict_prob_pairs = self.sort_bitstrings_by_probs(probs)

        # top_pairs = bitdict_prob_pairs[:50]  # Store top 50 solutions
        # problem.results = [
        #     SimpleNamespace(values=bit_dict, energy=0, frequency=1)
        #     for bit_dict, _ in top_pairs
        # ]

        problem.results = [SimpleNamespace(values=bit_dict, energy=0, frequency=1) for bit_dict, _ in bitdict_prob_pairs]
        sorted_probabilities = [p for _, p in bitdict_prob_pairs]

        return sorted_probabilities

