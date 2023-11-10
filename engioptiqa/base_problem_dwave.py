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
        if self.label_mapping is not None:
            self.binary_quadratic_model_indices = self.binary_quadratic_model.relabel_variables(
                self.label_mapping,inplace=False)
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
        
        # Only constraints:
        #self.poly = self.equilibrium_constraint_poly
        # print(model.to_qubo())

        #self.binary_quadratic_model = self.poly.compile().to_bqm()
        #print(self.binary_quadratic_model)
        
        # For the BQM, replace qubit names q[i][j] by indices 0,1,...
        self.label_mapping = {}
        self.label_mapping_inverse = {}
        enumerated_label = 0
        for i in range(self.rod.n_comp):
            for j in range(self.n_qubits_per_node):
                original_label = f"q[{i}][{j}]"
                self.label_mapping[original_label] = enumerated_label
                self.label_mapping_inverse[enumerated_label] = original_label
                enumerated_label += 1

        label_mapping_cs_inv = {}
        label_mapping_cs_inv_inverse = {}
        for i in range(self.rod.n_comp):
            original_label = f"q_A[{i}]"
            label_mapping_cs_inv[original_label] = enumerated_label
            label_mapping_cs_inv_inverse[enumerated_label] = original_label
            enumerated_label += 1

        self.label_mapping.update(label_mapping_cs_inv)
        self.label_mapping_inverse.update(label_mapping_cs_inv_inverse)

        # print(self.label_mapping)
        #print(self.poly)
        self.pyqubo_model = self.poly.compile()
        #print(self.pyqubo_model)
        self.binary_quadratic_model = self.pyqubo_model.to_bqm()
        # print(f'offset: {self.binary_quadratic_model.offset}')
        # self.binary_quadratic_model.offset = 0.0
        # print(f'offset: {self.binary_quadratic_model.offset}')
        self.binary_quadratic_model_indices = self.binary_quadratic_model.relabel_variables(self.label_mapping,inplace=False)
        # print("Original model:", self.binary_quadratic_model)
        # print("Model with indices:", self.binary_quadratic_model_indices)

    def visualize_qubo_matrix(self, show_fig=False, save_fig=False, suffix=''):
        
        title = self.name + '\n QUBO Matrix (PI + Manual Penalty) \n'
        # if hasattr(self, 'quad_method_name'):
            # title += self.quad_method_name

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

    def plot_qubo_matrix_pattern(self):
        title = self.name + '\n QUBO Pattern (PI + Manual Penalty) \n'
        # if hasattr(self, 'quad_method_name'):
            # title += self.quad_method_name
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
                filtered_variables = {key: value for key, value in result.items() if int(key[2]) == i_nf_poly}
                nf_sol.append(nf_poly_model.decode_sample(filtered_variables, vartype='Binary').energy)
            elif type(nf_poly) in [float, np.float64]:
                nf_sol.append(nf_poly)
            else:
                print(type(nf_poly))
                raise Exception('Unexpected type for nf_poly')  
        return nf_sol 
    
    def decode_cross_section_inverse_solution(self, result):

        cs_inv_sol = []       
        for cs_inv_poly in self.cs_inv_polys:
            # if isinstance(cs_inv_poly, Base):
            #     cs_inv_poly_model = cs_inv_poly.compile()

            if type(cs_inv_poly) in [float, np.float64]:
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