# from abc import abstractmethod
from amplify import QuadratizationMethod
import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import matplot2tikz
import sympy as sp

from .rod_1d import Rod1D
from .base_problem import BaseProblemRod1D

class DesignOptimizationProblemRod1D(BaseProblemRod1D):
    def __init__(self, rod, g, A_choice, output_path=None):
        super().__init__(rod, g, output_path)
        assert(len(A_choice)==2)
        self.A_choice = A_choice

        self.name = 'Design Optimization Problem'
        self.print_and_log(self.name+'\n')

    def get_analytical_cross_sections(self):

        L = self.rod.L
        A1 = self.A_choice[0]
        A2 = self.A_choice[1]

        opt_condition = sp.Eq((3*A1-A2)*self.x_sym**2 - 4*A1*L*self.x_sym + A1*L**2, 0)
        x_opt_val =sp.solve(opt_condition, self.x_sym)
        x_feasible = [x_val for x_val in x_opt_val if x_val > 0 and x_val <= L]
        if len(x_feasible) != 1:
            raise ValueError(f"Expected one valid solution for x in the range (0, {L}], but got \
                             {len(x_feasible)} valid solutions: {x_feasible}")
        x_opt = x_feasible[0]
        dx = L/self.rod.n_comp
        i_opt = round(x_opt/dx)

        A_opt = [A2 if i < i_opt else A1 for i in range(self.rod.n_comp)]

        return A_opt

    def compare_designs(self, solution):
        cs = [1./solution['cs_inv'][i] for i in range(self.rod.n_comp)]
        design_match = cs == list(self.A_analytic)
        output = f"Design match {design_match}\n"
        if not design_match:
            count = sum(x != y for x, y in zip(cs, list(self.A_analytic)))
            output += f"Number of mismatched components = {count}\n"
            output += f"Design (Solution)   = {cs}\n"
            output += f"Design (Analytical) = {list(self.A_analytic)}\n"
        return output

    def generate_discretization(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        BaseProblemRod1D.initialize_discretization(self)
        BaseProblemRod1D.generate_nodal_force_polys(self, n_qubits_per_var, binary_representation, lower_lim, upper_lim)
        self.generate_cross_section_inverse_polys()

    def generate_cross_section_inverse_polys(self):
        assert(self.variable_generator is not None)
        cs_inv_polys = []

        for _ in range(self.rod.n_comp):
            q = self.variable_generator.array("Binary", 1)
            cs_inv_polys.append(1./self.A_choice[0] + (1./self.A_choice[1]-1./self.A_choice[0])*q[0])
        self.cs_inv_polys = cs_inv_polys

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
            pos = (i_node+1)*self.n_qubits_per_var - 0.5
            plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.75)
            plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.75)

        pos = (self.rod.n_comp)*self.n_qubits_per_var + self.rod.n_comp - 0.5
        plt.axvline(x=pos, color='gray', linestyle='dotted', linewidth=0.75)
        plt.axhline(y=pos, color='gray', linestyle='dotted', linewidth=0.75)

    def visualize_qubo_matrix_sub_pattern(
            self, show_fig=False, save_fig=False, save_tikz=False, suffix='',
            highlight_cross_sections = False,
            highlight_interactions = False
        ):
        self.plot_qubo_matrix_pattern()
        self.annotate_qubo_matrix_pattern()
        plt.xlim(-0.5,((self.rod.n_comp)*(self.n_qubits_per_var+1)-0.5))
        plt.ylim(((self.rod.n_comp)*(self.n_qubits_per_var)),-0.5)

        if highlight_cross_sections:
            for i_comp in range(self.rod.n_comp-1):
                x_pos = self.rod.n_comp*self.n_qubits_per_var + i_comp - 0.5
                y_pos = i_comp*self.n_qubits_per_var - 0.5
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    1,
                    2*self.n_qubits_per_var,
                    linewidth = 2,
                    edgecolor='red',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

        if highlight_interactions:
            for i_comp in range(1,self.rod.n_comp):
                x_pos = self.rod.n_comp*self.n_qubits_per_var + (i_comp-1) - 0.5
                y_pos = i_comp*self.n_qubits_per_var - 0.5
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    2,
                    self.n_qubits_per_var,
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