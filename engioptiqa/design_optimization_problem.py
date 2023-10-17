from abc import abstractmethod
import itertools
from matplotlib import pyplot as plt
import numpy as np
import os

from .rod_1d import Rod1D
from .base_problem import BaseProblem

class DesignOptimizationProblem(BaseProblem):
    def __init__(self, rod, g, A_choice, output_path=None):
        super().__init__(rod, g, output_path)
        assert(len(A_choice)==2)
        self.A_choice = A_choice

        self.name = 'Design Optimization Problem'
        self.print_and_log(self.name+'\n')


    def analytical_complementary_energy_and_compliance(self):
        A_combi = list(itertools.product([self.A_choice[0], self.A_choice[1]], repeat=self.rod.n_comp))
        super().analytical_complementary_energy_and_compliance(A_combi)

    def get_optimal_solution(self):

        combi_opt = self.PI_combi.index(min(self.PI_combi))

        self.A_opt = self.A_combi[combi_opt]
        self.PI_opt = self.PI_combi[combi_opt]
        self.C_opt = self.C_combi[combi_opt]

        output = 'Optimal Solution:\n'
        output+= f'\tCross Section: {self.A_opt}\n'
        output+= f'\tComplementary Energy: {self.PI_opt}\n'
        output+= f'\tCompliance: {self.C_opt}\n'
        self.print_and_log(output)

    def compute_analytical_solution(self):
        self.analytical_complementary_energy_and_compliance()
        self.get_optimal_solution()

        self.A_analytic = self.A_opt
        self.PI_analytic = self.PI_opt
        self.C_analytic = self.C_opt
        
        self.rod_1d_opt = Rod1D(self.rod.n_comp,self.rod.L,0)
        self.rod_1d_opt.cross_sections = self.A_opt
        if self.output_path is not None:
            file_name = os.path.join(self.output_path,'rod_optimal.png')
        else:
            file_name = None
        self.rod_1d_opt.visualize(file_name, self.save_fig)
        self.stress_analytic = self.compute_stress_function(self.rod_1d_opt)
        self.force_analytic = self.compute_force_function(self.stress_analytic, self.rod_1d_opt)
        self.displacement_analytic = self.compute_displacement_function(self.stress_analytic, self.rod_1d_opt)

        output = f'Analytic Force: {self.force_analytic}\n'
        self.print_and_log(output)

    @abstractmethod
    def generate_cross_section_polys(self):
        pass

    def visualize_qubo_matrix_pattern(self, show_fig=False, save_fig=False, suffix=''):
        super().plot_qubo_matrix_pattern()
        self.annotate_qubo_matrix_pattern()
        if show_fig:
            plt.show()
        if save_fig:
            super().save_qubo_matrix_pattern(suffix)
        plt.close()

    def annotate_qubo_matrix_pattern(self):
        for i_node in range(self.rod.n_comp):
            pos = (i_node+1)*self.n_qubits_per_node - 0.5
            plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.75)
            plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.75)
        
        pos = (self.rod.n_comp)*self.n_qubits_per_node + self.rod.n_comp - 0.5
        plt.axvline(x=pos, color='gray', linestyle='dotted', linewidth=0.75)
        plt.axhline(y=pos, color='gray', linestyle='dotted', linewidth=0.75)

    def visualize_qubo_matrix_sub_pattern(self, save_fig=False, suffix=''):
        super().plot_qubo_matrix_pattern()
        self.annotate_qubo_matrix_pattern()
        plt.xlim(-0.5,((self.rod.n_comp)*(self.n_qubits_per_node+1)-1))
        plt.ylim(((self.rod.n_comp)*(self.n_qubits_per_node+1)-1),-0.5)
        if save_fig:
            self.save_qubo_matrix_sub_pattern(suffix)
        plt.close()

    def save_qubo_matrix_sub_pattern(self, suffix=''):
        file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_QUBO_sub_pattern' + suffix)
        plt.savefig(file_name, dpi=600)