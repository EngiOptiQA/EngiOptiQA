from abc import ABC, abstractmethod
from amplify import AcceptableDegrees, VariableGenerator
from dimod import BinaryQuadraticModel as BinaryQuadraticModelDWave
from dimod.views.samples import SampleView
from dimod.sampleset import SampleSet
import matplot2tikz
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

class Problem(ABC):
    """
    Abstract base class for EngiOptiQA problems.

    :param output_path: Optional path for saving results.
    """
    def __init__(self, output_path = None):
        if output_path is None:
            self.save_fig = False
        else:
            self.save_fig = True
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                print(f"Folder '{output_path}' created successfully.")
            else:
                print(f"Folder '{output_path}' already exists.")
            self.log_file = os.path.join(output_path,'log.txt')
        self.output_path = output_path

        self.name = 'EngiOptiQA Problem'

    # I/O
    # ---
    def set_output_path(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Folder '{output_path}' created successfully.")
        else:
            print(f"Folder '{output_path}' already exists.")
        self.log_file = os.path.join(output_path,'log.txt')
        self.output_path = output_path

        self.print_and_log(self.name+'\n')

    def print_and_log(self, output):
        print(output)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as file:
                file.write(output)

    # Discretization using binary variables
    # -------------------------------------
    def initialize_discretization(self):
        self.variable_generator = VariableGenerator()

    @abstractmethod
    def generate_discretization(self):
        pass

    # Support for adaptive encoding of continuous variables
    # -----------------------------------------------------
    def get_number_of_continuous_vars(self):
        return 0

    def update_ranges(self, sol_bit_array, sol, sol_prev, relaxation_factor, verbose=False):

        n_vars = self.get_number_of_continuous_vars()
        if n_vars == 0:
            raise Exception('Attempt to update ranges, but number of continuous variables is zero.')
        actions = ['' for _ in range(n_vars)]
        sol_decoded = [np.nan for _ in range(n_vars)]
        sol_decoded_new = [np.nan for _ in range(n_vars)]
        sol_encoded = [[] for _ in range(n_vars)]
        sol_encoded_new = [[] for _ in range(n_vars)]
        for i_var in range(n_vars):

            # Extract bit array for current node
            start = i_var * self.n_qubits_per_var
            end = (i_var + 1) * self.n_qubits_per_var
            sol_encoded[i_var] = sol_bit_array[start:end]

            sol_decoded[i_var] = self.real_number.decode_bits_to_real(
                sol_encoded[i_var], self.a_min[i_var], self.a_max[i_var])

            self.a_min[i_var], self.a_max[i_var], actions[i_var] = self.real_number.update_range(
                sol[i_var], sol_encoded[i_var], sol_prev[i_var], self.a_min[i_var], self.a_max[i_var],
                relaxation_factor, verbose
            )

            sol_encoded_new[i_var] = self.real_number.encode_real_to_bits(
                sol[i_var], self.a_min[i_var], self.a_max[i_var])

            sol_decoded_new[i_var] = self.real_number.decode_bits_to_real(
                sol_encoded_new[i_var], self.a_min[i_var], self.a_max[i_var]
            )

        return actions, sol_decoded, sol_decoded_new, sol_encoded, sol_encoded_new

    def update_solution(self, sol_bit_array, sol_encoded):
        n_vars = self.get_number_of_continuous_vars()
        for i_var in range(n_vars):
            start = i_var * self.n_qubits_per_var
            end = (i_var + 1) * self.n_qubits_per_var
            sol_bit_array[start:end] = sol_encoded[i_var]
        return sol_bit_array

    # Problem formulation, i.e., model providing polynomial in binary variables
    # -------------------------------------------------------------------------
    @abstractmethod
    def generate_problem_formulation(self):
        pass

    # QUBO
    # ----
    def transform_to_dwave(self):

        bq = AcceptableDegrees(objective={"Binary": "Quadratic"})
        im, mapping =  self.binary_model.to_intermediate_model(bq, quadratization_method="IshikawaKZFD")

        output = f'Number of binary variables (original degree): {len(self.binary_model.get_variables())}\n'
        output+= f'Number of binary variables (reduced degree) : {len(im.get_variables())}\n'
        self.print_and_log(output)
        coeff_dict = im.objective.asdict()
        constant = coeff_dict.get((), 0.0)
        linear = {k[0]: v for k, v in coeff_dict.items() if len(k) == 1}
        quadratic = {tuple(k): v for k, v in coeff_dict.items() if len(k) == 2}

        self.binary_quadratic_model_dwave = BinaryQuadraticModelDWave(linear, quadratic, constant, vartype='BINARY')

    def get_qubo_matrix(self):
        bq = AcceptableDegrees(objective={"Binary": "Quadratic"})
        im, mapping =  self.binary_model.to_intermediate_model(bq, quadratization_method="IshikawaKZFD")
        coeff_dict = im.objective.asdict()

        # 1. Determine the number of variables
        variable_keys = [k for k in coeff_dict.keys() if k]  # remove empty tuple
        n = max(max(k) for k in variable_keys) + 1 if variable_keys else 0

        # 2. Initialize an NxN matrix of zeros
        Q = np.zeros((n, n))

        # 3. Fill the matrix
        for key, value in coeff_dict.items():
            if key == ():
                # constant offset, ignore in the matrix
                continue
            elif len(key) == 1:
                # Linear term
                i = key[0]
                Q[i, i] = value
            elif len(key) == 2:
                # Quadratic term
                i, j = key
                Q[i, j] = value
                Q[j, i] = value  # make it symmetric
            else:
                raise ValueError(f"Unexpected key format in dictionary with QUBO coefficients: {key}")

        return Q

    def visualize_qubo_matrix(self, show_fig=False, save_fig=False, save_tikz=False, suffix=''):
        title = self.name + '\n QUBO Matrix \n'

        print("Generating QUBO matrix for visualization...")

        Q = self.get_qubo_matrix()

        # Visualize the QUBO Matrix.
        plt.figure()
        plt.suptitle(title)
        plt.imshow(Q,interpolation='none')
        plt.colorbar()
        if show_fig:
            plt.show()
        if save_fig or save_tikz:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_qubo_matrix' + suffix)
            if save_fig:
                plt.savefig(file_name, dpi=600)
            if save_tikz:
                matplot2tikz.save(file_name + '.tex')
        plt.close()

    # Post-processing of results
    # --------------------------
    def get_bit_array(self, result):
        if type(result) is SampleView:
            bit_array = [int(result[k]) for k in result.keys()]
        else:
            bit_array = [int(result.values[k]) for k in result.values.keys()]
        return bit_array

    def get_energy(self, index):
        if type(self.results) is SampleSet:
            return self.results.record[index]['energy']
        else:
            if hasattr(self.results[index], 'energy'):
                return self.results[index].energy
            else:
                return np.nan

    def get_frequency(self, index):
        if type(self.results) is SampleSet:
            return self.results.record[index]['num_occurrences']
        else:
            if hasattr(self.results[index], 'frequency'):
                return self.results[index].frequency
            else:
                return np.nan

    def analyze_results(self, results=None, analysis_plots=True, compute_errors=True, result_max=sys.maxsize):
        if results is None and not hasattr(self, 'results'):
            raise Exception('Attempt to analyze results, but no results exist or have been passed.')
        elif results is None and hasattr(self, 'results'):
            results = self.results

        solutions = [{} for _ in range(len(results))]
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)

            solutions[i_result]['bit_array'] = bit_array
            solutions[i_result]['energy'] = self.get_energy(i_result)
            solutions[i_result]['frequency'] = self.get_frequency(i_result)

        return solutions

    def decode_amplify_poly_with_bitstring(self, amplify_poly, bitstring):

        poly_dict = amplify_poly.as_dict()
        value = 0.0
        for vars_tuple, coeff in poly_dict.items():
            if len(vars_tuple) == 0:
                # Constant term
                term_value = coeff
            else:
                # Product of variables in the tuple
                term_value = coeff * np.prod([bitstring[i] for i in vars_tuple])
            value += term_value
        return float(value)