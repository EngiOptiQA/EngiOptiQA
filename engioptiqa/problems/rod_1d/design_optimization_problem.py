# from abc import abstractmethod
from amplify import QuadratizationMethod
import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import matplot2tikz

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

    def generate_discretization(self, n_qubits_per_node, binary_representation, lower_lim=None, upper_lim=None):
        BaseProblem.initialize_discretization(self)
        BaseProblem.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation, lower_lim, upper_lim)
        self.generate_cross_section_inverse_polys()

    def generate_cross_section_inverse_polys(self):
        assert(self.variable_generator is not None)
        cs_inv_polys = []

        for _ in range(self.rod.n_comp):
            q = self.variable_generator.array("Binary", 1)
            cs_inv_polys.append(1./self.A_choice[0] + (1./self.A_choice[1]-1./self.A_choice[0])*q[0])
        self.cs_inv_polys = cs_inv_polys

    def set_quad_method(self,quad_method_name):
        self.quad_method_name = quad_method_name
        match quad_method_name:
            case 'ISHIKAWA':
                self.quad_method = QuadratizationMethod.ISHIKAWA
            case 'ISHIKAWA_KZFD':
                self.quad_method = QuadratizationMethod.ISHIKAWA_KZFD
            case 'SUBSTITUTION':
                self.quad_method = QuadratizationMethod.SUBSTITUTION
            case 'SUBSTITUTION_KZFD':
                self.quad_method = QuadratizationMethod.SUBSTITUTION_KZFD
            case _ :
                raise Exception('Unknown quad_method', quad_method_name)

    def visualize_qubo_matrix_pattern(self, show_fig=False, save_fig=False, save_tikz=False, suffix=''):
        self.plot_qubo_matrix_pattern()
        self.annotate_qubo_matrix_pattern()
        if show_fig:
            plt.show()
        if save_fig or save_tikz:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_QUBO_pattern' + suffix)
            if save_fig:
                plt.savefig(file_name, dpi=600)
            if save_tikz:
                matplot2tikz.save(file_name + '.tex')
        plt.close()

    def annotate_qubo_matrix_pattern(self):
        for i_node in range(self.rod.n_comp):
            pos = (i_node+1)*self.n_qubits_per_node - 0.5
            plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.75)
            plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.75)

        pos = (self.rod.n_comp)*self.n_qubits_per_node + self.rod.n_comp - 0.5
        plt.axvline(x=pos, color='gray', linestyle='dotted', linewidth=0.75)
        plt.axhline(y=pos, color='gray', linestyle='dotted', linewidth=0.75)

    def visualize_qubo_matrix_sub_pattern(
            self, show_fig=False, save_fig=False, save_tikz=False, suffix='',
            highlight_cross_sections = False,
            highlight_interactions = False
        ):
        self.plot_qubo_matrix_pattern()
        self.annotate_qubo_matrix_pattern()
        plt.xlim(-0.5,((self.rod.n_comp)*(self.n_qubits_per_node+1)-0.5))
        plt.ylim(((self.rod.n_comp)*(self.n_qubits_per_node)),-0.5)

        if highlight_cross_sections:
            for i_comp in range(self.rod.n_comp-1):
                x_pos = self.rod.n_comp*self.n_qubits_per_node + i_comp - 0.5
                y_pos = i_comp*self.n_qubits_per_node - 0.5
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    1,
                    2*self.n_qubits_per_node,
                    linewidth = 2,
                    edgecolor='red',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

        if highlight_interactions:
            for i_comp in range(1,self.rod.n_comp):
                x_pos = self.rod.n_comp*self.n_qubits_per_node + (i_comp-1) - 0.5
                y_pos = i_comp*self.n_qubits_per_node - 0.5
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    2,
                    self.n_qubits_per_node,
                    linewidth = 2,
                    edgecolor='orange',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

        if show_fig:
            plt.show()
        if save_fig or save_tikz:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_QUBO_sub_pattern' + suffix)
            if highlight_cross_sections:
                suffix += '_highlight_cross_sections'
            if highlight_interactions:
                suffix += '_highlight_interactions'
            if save_fig:
                plt.savefig(file_name, dpi=600)
            if save_tikz:
                matplot2tikz.save(file_name + '.tex')
        plt.close()