from amplify import (
    BinaryPoly, 
    BinaryQuadraticModel,
    save_lp,
    SymbolGenerator)

from dimod import BinaryQuadraticModel as BinaryQuadraticModelDWave
from dimod import cqm_to_bqm, lp
from dimod.views.samples import SampleView
from dimod.sampleset import SampleSet

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle
import tikzplotlib

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

        # TODO Scaling
        # PI_abs = np.abs(self.PI_qubo_matrix.to_numpy())
        # PI_max = np.max(PI_abs)
        # print("Magnitude Complementary Energy", PI_max)

        # con_eq_abs = np.abs(self.constraints_qubo_matrix.to_numpy())
        # con_eq_max = np.max(con_eq_abs)
        # print("Magnitude Constraint EQ", con_eq_max)

        # Options for penalty weight:
        # 1. Scale
        # self.penalty_weight_equilibrium = PI_max/con_eq_max * penalty_weight
        # 2. Do not scale
        self.penalty_weight_equilibrium = penalty_weight
        print(f"Effective penalty weight: {self.penalty_weight_equilibrium}\n")
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_equilibrium * self.equilibrium_constraint_poly

        if self.quad_method is not None:
            print(self.quad_method)
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly, method=self.quad_method)
        else:
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly)

        self.qubo_matrix, self.PI_QUBO_const = self.binary_quadratic_model.logical_matrix

        output = f'Number of input qubits: {self.binary_quadratic_model.num_input_vars}\n'
        output+= f'Number of logical qubits: {self.binary_quadratic_model.num_logical_vars}\n'
        self.print_and_log(output)

    def update_penalty_weight_in_qubo_formulation(self, penalty_weight = 1.0):
        self.penalty_weight_equilibrium = penalty_weight
        print(f"Effective penalty weight: {self.penalty_weight_equilibrium}\n")
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_equilibrium * self.equilibrium_constraint_poly
        if self.quad_method is not None:
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly, method=self.quad_method)
        else:
            self.binary_quadratic_model = BinaryQuadraticModel(self.poly)
        
    def visualize_qubo_matrix(self, show_fig=False, save_fig=False, save_tikz=False, suffix=''):
        title = self.name + '\n QUBO Matrix \n'

        # Visualize the QUBO Matrix.
        plt.figure()
        plt.suptitle(title)
        plt.imshow(self.qubo_matrix.to_numpy(),interpolation='none')
        plt.colorbar()
        if show_fig:
            plt.show()
        if save_fig or save_tikz:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_qubo_matrix' + suffix)
            if save_fig:
                plt.savefig(file_name, dpi=600)
            if save_tikz:
                tikzplotlib.save(file_name + '.tex')
        plt.close()

    def plot_qubo_matrix_pattern(self, highlight_nodes=False, highlight_interactions=False):
        title = self.name + '\n QUBO Pattern \n'
        binary_matrix = np.where(self.qubo_matrix.to_numpy() != 0, 1, 0)
        plt.figure()
        plt.suptitle(title)
        plt.imshow(binary_matrix,cmap='gray_r')

        if highlight_nodes:
            for i_node in range(self.rod.n_comp):
                x_pos = (i_node)*self.n_qubits_per_node - 0.5
                y_pos = x_pos
                rect = patches.Rectangle(
                    (x_pos,y_pos), 
                    self.n_qubits_per_node, 
                    self.n_qubits_per_node,
                    linewidth = 2,
                    edgecolor='red', 
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

        if highlight_interactions:
            for i_node in range(self.rod.n_comp-1):
                x_pos = (i_node)*self.n_qubits_per_node - 0.5
                y_pos = x_pos
                rect = patches.Rectangle(
                    (x_pos,y_pos), 
                    2*self.n_qubits_per_node, 
                    2*self.n_qubits_per_node,
                    linewidth = 2,
                    edgecolor='orange', 
                    facecolor='none'
                )
                plt.gca().add_patch(rect)        

    def transform_to_dwave(self, lp_file_path):
        save_lp(self.binary_quadratic_model, lp_file_path)
        # Import as DWave constrained quadratic model (CQM)
        cqm = lp.load(lp_file_path)
        # Transform to DWave BQM
        bqm = cqm_to_bqm(cqm)
        # Set BQM problem that is solved by AnnealingSolverDWave
        self.binary_quadratic_model_indices = BinaryQuadraticModelDWave(bqm[0])

        mapping_x_to_i = {}
        for i_var in range(self.binary_quadratic_model_indices.num_variables):  
            mapping_x_to_i.update({f'x_{i_var}': i_var})
        self.binary_quadratic_model_indices.relabel_variables(mapping_x_to_i, inplace=True)

    def decode_nodal_force_solution(self, result):
        nf_sol = []
        for nf_poly in self.nf_polys:
            if type(nf_poly) is BinaryPoly:
                if type(result) is SampleView:
                    result_tmp = [int(x) for x in result._data]
                    nf_sol.append(nf_poly.decode(result_tmp))
                else:
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
                if type(result) is SampleView:
                    result_tmp = [int(x) for x in result._data]
                    cs_inv_sol.append(cs_inv_poly.decode(result_tmp))
                else:
                    cs_inv_sol.append(cs_inv_poly.decode(result.values))   
            elif type(cs_inv_poly) in [float, np.float64]:
                cs_inv_sol.append(cs_inv_poly)
            else:
                print(type(cs_inv_poly))
                raise Exception('Unexpected type for cs_inv_poly')
        return cs_inv_sol
    
    def get_energy(self, index):
        if type(self.results) is SampleSet:
            return self.results.record[index]['energy']
        else:
            return self.results[index].energy
    
    def get_frequency(self, index):
        if type(self.results) is SampleSet:
            return self.results.record[index]['num_occurrences']
        else:
            return self.results[index].frequency

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