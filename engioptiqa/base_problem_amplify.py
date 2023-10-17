from amplify import (
    BinaryPoly, 
    BinaryQuadraticModel,
    SymbolGenerator)
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from .base_problem import BaseProblem
from .real_number import RealNumber
from .solution_emulator import SolutionEmulator

class BaseProblemAmplify(BaseProblem):

    def initialize_discretization(self):
        self.symbol_generator = SymbolGenerator(BinaryPoly)

    def generate_nodal_force_polys(self, n_qubits_per_node, binary_representation):
        assert(self.symbol_generator is not None)
        self.n_qubits_per_node = n_qubits_per_node
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_node, self.binary_representation)
        nf_polys = []

        for i_comp in range(self.rod.n_comp):
            q = self.symbol_generator.array(self.n_qubits_per_node)
            nf_polys.append(self.real_number.evaluate(q))
            if i_comp == self.rod.n_comp-1:
                nf_polys.append(0.0)
        self.nf_polys = nf_polys            
        
    def generate_qubo_formulation(self, penalty_weight=1):
        self.generate_complementary_energy_poly()
        self.generate_constraint_polys()

        if self.quad_method is not None:
            PI_quadratic_model = BinaryQuadraticModel(self.complementary_energy_poly, method=self.quad_method)
            constraints_quadratic_model = BinaryQuadraticModel(self.equilibrium_constraint_poly, method=self.quad_method)
        else:
            PI_quadratic_model = BinaryQuadraticModel(self.complementary_energy_poly)
            constraints_quadratic_model = BinaryQuadraticModel(self.equilibrium_constraint_poly)

        self.PI_qubo_matrix, self.PI_QUBO_const = PI_quadratic_model.logical_matrix
        self.constraints_qubo_matrix, self.constraints_QUBO_const = constraints_quadratic_model.logical_matrix

        PI_abs = np.abs(self.PI_qubo_matrix.to_numpy())
        PI_max = np.max(PI_abs)
        #print("Magnitude Complementary Energy", PI_max)

        con_eq_abs = np.abs(self.constraints_qubo_matrix.to_numpy())
        con_eq_max = np.max(con_eq_abs)
        #print("Magnitude Constraint EQ", con_eq_max)

        self.penalty_weight_equilibrium = PI_max/con_eq_max * penalty_weight
        # self.penalty_weight_equilibrium = penalty_weight

        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_equilibrium * self.equilibrium_constraint_poly

        # print(self.poly)
        if self.quad_method is not None:
            print(self.quad_method)
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly, method=self.quad_method)
        else:
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly)

        self.qubo_matrix, self.PI_QUBO_const = self.binary_quadratic_model.logical_matrix

        output = f'Number of input qubits: {self.binary_quadratic_model.num_input_vars}\n'
        output+= f'Number of logical qubits: {self.binary_quadratic_model.num_logical_vars}\n'
        self.print_and_log(output)


    def visualize_qubo_matrix(self, show_fig=False, save_fig=False, suffix=''):
        title = self.name + '\n QUBO Matrix (PI + Manual Penalty) \n'
        # if hasattr(self, 'quad_method_name'):
            # title += self.quad_method_name

        # Visualize the QUBO Matrix.
        plt.figure()
        plt.suptitle(title)
        plt.imshow(self.qubo_matrix.to_numpy(),interpolation='none')
        plt.colorbar()
        if show_fig:
            plt.show()
        if save_fig:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_qubo_matrix' + suffix)
            plt.savefig(file_name, dpi=600)
        plt.close()

    def plot_qubo_matrix_pattern(self):
        title = self.name + '\n QUBO Pattern (PI + Manual Penalty) \n'
        # if hasattr(self, 'quad_method_name'):
            # title += self.quad_method_name
        binary_matrix = np.where(self.qubo_matrix.to_numpy() != 0, 1, 0)
        plt.figure()
        plt.suptitle(title)
        plt.imshow(binary_matrix,cmap='gray_r')

    def decode_nodal_force_solution(self, result):
        nf_sol = []
        for nf_poly in self.nf_polys:
            if type(nf_poly) is BinaryPoly:
                nf_sol.append(nf_poly.decode(result.values))
            elif type(nf_poly) in [float, np.float64]:
                nf_sol.append(nf_poly)
            else:
                print(type(nf_poly))
                raise Exception('Unexpected type for nf_poly')  
        return nf_sol      

    def decode_cross_section_inverse_solution(self, result):

        cs_inv_sol = []       
        for cs_inv_poly in self.cs_inv_polys:
            if type(cs_inv_poly) is BinaryPoly:
                cs_inv_sol.append(cs_inv_poly.decode(result.values))   
            elif type(cs_inv_poly) in [float, np.float64]:
                cs_inv_sol.append(cs_inv_poly)
            else:
                print(type(cs_inv_poly))
                raise Exception('Unexpected type for cs_inv_poly')   
        return cs_inv_sol
    
    def store_results(self):
        if hasattr(self, 'results'):
            # Results is of class amplify.SolverResult and stores a list of solutions.
            # These are of class amplify.SolverSolution and are converted to SolutionEmulator for storing.
            results = []
            for result in self.results:
                results.append(
                    SolutionEmulator(
                        energy=result.energy, 
                        frequency=result.frequency,
                        is_feasible=result.is_feasible,
                        values=result.values
                    )
                )
            # Store results, i.e., a list of SolutionEmulator objects, each reflecting one solution.
            results_file = os.path.join(self.output_path, 'results.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
        else:
            raise Exception('Trying to store results but no results exist.') 

    def load_results(self, results_file):
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)    