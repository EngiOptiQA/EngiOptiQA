from matplotlib import pyplot as plt
import numpy as np
import os
from pyqubo import Array, Base

from .base_problem import BaseProblem
from .real_number import RealNumber

class BaseProblemDWave(BaseProblem):

    def generate_nodal_force_polys(self, n_qubits_per_node, binary_representation):
        self.n_qubits_per_node = n_qubits_per_node
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_node, self.binary_representation)
        nf_polys = []

        q = Array.create('q', shape=(self.rod.n_comp, n_qubits_per_node), vartype='BINARY')
        for i_comp in range(self.rod.n_comp):
            nf_polys.append(self.real_number.evaluate(q[i_comp]))
            if i_comp == self.rod.n_comp-1:
                nf_polys.append(0.0)

        self.nf_polys = nf_polys      

    def update_penalty_weight_in_qubo_formulation(self, penalty_weight = 1.0):
        self.penalty_weight_equilibrium = penalty_weight
        self.poly = self.complementary_energy_poly + \
             self.penalty_weight_equilibrium * self.equilibrium_constraint_poly
        self.pyqubo_model = self.poly.compile()
        self.binary_quadratic_model = self.pyqubo_model.to_bqm()
        if self.mapping_q_to_i is not None:
            self.binary_quadratic_model_indices = self.binary_quadratic_model.relabel_variables(
                self.mapping_q_to_i, inplace=False)
        else:
            raise Exception('Cannot update QUBO formulation without label mapping.')

    def generate_qubo_formulation(self, penalty_weight = 1.0):

        self.generate_complementary_energy_poly()
        self.generate_constraint_polys()

        self.penalty_weight_equilibrium = penalty_weight

        self.bqm_objective = self.complementary_energy_poly.compile().to_bqm()
        self.bqm_constraints = self.equilibrium_constraint_poly.compile().to_bqm() 

        self.poly = self.complementary_energy_poly + \
             self.penalty_weight_equilibrium * self.equilibrium_constraint_poly
                
        # Two options for variables in BQM
        # 1.) q[i][j] and q_A[i] => Problem: cannot be solved 
        # 2.) indices => Problem: for design optimization problem (cubic), results cannot be analyzed

        # For the BQM, replace qubit names q[i][j] by indices 0,1,...
        self.mapping_q_to_i = {}
        self.mapping_i_to_q = {}
        enumerated_label = 0
        for i in range(self.rod.n_comp):
             for j in range(self.n_qubits_per_node):
                 original_label = f"q[{i}][{j}]"
                 self.mapping_q_to_i[original_label] = enumerated_label
                 self.mapping_i_to_q[enumerated_label] = original_label
                 enumerated_label += 1

        # For the BQM of the design optimization problem, replace qubit names q_A[i] by indices
        # label_mapping_cs_inv = {}
        # label_mapping_cs_inv_inverse = {}
        # for i in range(self.rod.n_comp):
        #     original_label = f"q_A[{i}]"
        #     label_mapping_cs_inv[original_label] = enumerated_label
        #     label_mapping_cs_inv_inverse[enumerated_label] = original_label
        #     enumerated_label += 1
        # self.label_mapping.update(label_mapping_cs_inv)
        # self.label_mapping_inverse.update(label_mapping_cs_inv_inverse)

        self.pyqubo_model = self.poly.compile()
        self.binary_quadratic_model = self.pyqubo_model.to_bqm()       

        self.binary_quadratic_model_indices = self.pyqubo_model.to_bqm(index_label=True)
        
        print("Original model:", self.binary_quadratic_model)
        print("Model with indices:", self.binary_quadratic_model_indices)

    def visualize_qubo_matrix(self, show_fig=False, save_fig=False, suffix=''):
        
        title = self.name + '\n QUBO Matrix \n'

        # Visualize the QUBO Matrix.
        plt.figure()
        plt.suptitle(title)
        plt.imshow(self.binary_quadratic_model_indices.to_numpy_matrix(),interpolation='none')
        plt.colorbar()
        if show_fig:
            plt.show()
        if save_fig:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_qubo_matrix' + suffix)
            plt.savefig(file_name, dpi=600)
        plt.close()

    def plot_qubo_matrix_pattern(self, highlight_nodes = False, highlight_interactions=False):
        title = self.name + '\n QUBO Pattern \n'
        binary_matrix = np.where(self.binary_quadratic_model_indices.to_numpy_matrix() != 0, 1, 0)
        plt.figure()
        plt.suptitle(title)
        plt.imshow(binary_matrix,cmap='gray_r')

    def decode_nodal_force_solution(self, result):

        nf_sol = []
        for i_nf_poly, nf_poly in enumerate(self.nf_polys):
            if isinstance(nf_poly, Base):
                nf_poly_model = nf_poly.compile()
                # Filter symbolic variables for i-th nf_poly,
                # which only contains symbolic variables q[i][j] (j=1,...,n_qubits_per_node)
                
                # TODO Design Optimization Problem DWave
                # Additionally filter out q_A[i] 
                #for key, value in result.items():
                #    if (not '*' in key) and (not '_' in key) and (int(key[2]) == i_nf_poly):
                #        filtered_variables = {key: value}

                filtered_variables = {key: value for key, value in result.items() if int(key[2]) == i_nf_poly}
                nf_sol.append(nf_poly_model.decode_sample(filtered_variables, vartype='Binary').energy)

                # Problem: result contains indices only, nf_poly still q[i][j] etc.
                #nf_sol.append(nf_poly_model.decode_sample(result, vartype='Binary').energy)

            elif type(nf_poly) in [float, np.float64]:
                nf_sol.append(nf_poly)
            else:
                print(type(nf_poly))
                raise Exception('Unexpected type for nf_poly')  
        return nf_sol 
    
    def decode_cross_section_inverse_solution(self, result):

        cs_inv_sol = []       
        for i_cs_inv_poly, cs_inv_poly in enumerate(self.cs_inv_polys):
            if isinstance(cs_inv_poly, Base):
                cs_inv_poly_model = cs_inv_poly.compile()

                # TODO Design Optimization Problem DWave
                # Filter out variables

                #for key, value in result.items():
                #    if ('q_A' in key) and (not '*' in key): 
                #        if int(key[4]) == i_cs_inv_poly:
                #            filtered_variables = {key: value}
                #cs_inv_sol.append(cs_inv_poly_model.decode_sample(filtered_variables, vartype='Binary').energy)

                cs_inv_sol.append(cs_inv_poly_model.decode_sample(result, vartype='Binary').energy)
               
            elif type(cs_inv_poly) in [float, np.float64]:
                cs_inv_sol.append(cs_inv_poly)
            else:
                print(type(cs_inv_poly))
                raise Exception('Unexpected type for cs_inv_poly')   
        return cs_inv_sol
    
    def get_energy(self, index):
        return self.results.record[index]['energy']
    
    def get_frequency(self, index):
        return self.results.record[index]['num_occurrences']
    
    def store_results(self):
        raise Exception('store_results not yet implemented for BaseProblemDWave')

    def load_results(self, results_file):
        raise Exception('load_results not yet implemented for BaseProblemDWave')