from abc import ABC, abstractmethod
from amplify import (
    AcceptableDegrees,
    BinaryQuadraticModel,
    Model,
    Poly,
    VariableGenerator)
from dimod import BinaryQuadraticModel as BinaryQuadraticModelDWave
from dimod import cqm_to_bqm, lp
from dimod.views.samples import SampleView
from dimod.sampleset import SampleSet
import math
import matplot2tikz
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from prettytable import PrettyTable
from scipy.integrate import quad
import sympy as sp
import sys

from engioptiqa.variables.real_number import RealNumber
from engioptiqa.problems.problem import Problem
from .rod_1d import Rod1D

from types import SimpleNamespace

class BaseProblemRod1D(Problem):
    def __init__(self, rod, g, output_path=None):
        super().__init__(output_path)
        self.name = 'Base Problem Rod 1D'
        self.rod = rod
        self.g = g

        self.x_sym = sp.symbols('x')


    def capabilities(self):
        return super().capabilities() | {"outeropt_penalty", "outeropt_augmented_lagrangian"}

    def analytical_complementary_energy_and_compliance(self, cross_sections):

        n_comp = self.rod.n_comp
        x = self.rod.x
        E = self.rod.E
        rho = self.rod.rho
        g = self.g

        A = cross_sections
        PI = 0.0
        C  = 0.0
        tmp_rod_1d = Rod1D(n_comp, self.rod.L, A)

        # Stress
        stress = self.compute_stress_function(tmp_rod_1d)

        # Displacement
        u = self.compute_displacement_function(stress, tmp_rod_1d)

        # Complementary Energy
        PI_elem = []
        for i_comp in range(n_comp):
            expr = A[i_comp]/E[i_comp] * stress[i_comp]**2
            PI_elem.append(1./2. * sp.integrate(expr,(self.x_sym, x[i_comp], x[i_comp+1])))
        PI = sum(PI_elem)

        # Compliance
        C_elem = []
        for i_comp in range(n_comp):
            vol_force = rho[i_comp]*g
            expr = stress[i_comp]/E[i_comp]
            C_elem.append(A[i_comp]*sp.integrate(vol_force*u[i_comp], (self.x_sym, x[i_comp], x[i_comp+1])))
        C = sum(C_elem)

        # Sanity check.
        assert(math.isclose(PI, C/2.0, rel_tol=1e-9))

        return PI, C

    def compute_stress_function(self, rod):

        n_comp = rod.n_comp
        x = rod.x
        cs = rod.cross_sections
        rho = rod.rho

        g = self.g

        stress = []
        stress.append(rho[-1]*g*(x[-1]-self.x_sym))
        for i_comp in range(n_comp-2, -1, -1):
            stress.append(cs[i_comp+1]/cs[i_comp]*stress[-1].subs(self.x_sym, x[i_comp+1]) + rho[i_comp]*g*(x[i_comp+1]-self.x_sym))
        stress.reverse()

        return stress

    def compute_force_function(self, stress, rod):
        n_comp = rod.n_comp
        cs = rod.cross_sections

        force = []
        for i_comp in range(n_comp):
            force.append(stress[i_comp]*cs[i_comp])

        return force

    def compute_displacement_function(self, stress, rod):

        n_comp = rod.n_comp
        x = rod.x
        E = rod.E

        u = []
        u.append(sp.integrate(stress[0]/E[0], self.x_sym))
        for i_comp in range(1, n_comp):
            expr = stress[i_comp]/E[i_comp]
            u.append(u[-1].subs(self.x_sym, x[i_comp]) + sp.integrate(expr, (self.x_sym, x[i_comp], self.x_sym)))

        return u

    def compute_analytical_solution(self):
        output = "Analytical Solution:\n"
        self.A_analytic = self.get_analytical_cross_sections()
        output += f"\tCross Sections: {self.A_analytic}\n"
        self.PI_analytic, self.C_analytic = self.analytical_complementary_energy_and_compliance(self.A_analytic)
        output += f"\tComplementary Energy: {self.PI_analytic}\n\tCompliance: {self.C_analytic}\n"
        self.rod.cross_sections = self.A_analytic
        self.stress_analytic = self.compute_stress_function(self.rod)
        self.force_analytic = self.compute_force_function(self.stress_analytic, self.rod)
        self.displacement_analytic = self.compute_displacement_function(self.stress_analytic, self.rod)
        output += f'\tForce: {self.force_analytic}\n'
        self.print_and_log(output)

    # Generate Basis Functions.
    def basis(self, xi, xj, x_sym):
        phi1 = (xj - x_sym)/(xj-xi)
        phi2 = (x_sym-xi)/(xj-xi)
        return phi1, phi2

    def get_real_number_object(self, i_group):
        assert(i_group == 0)
        return self.real_number

    def has_adaptive_variables(self):
        return self.binary_representation == 'adaptive_range'

    def get_number_of_adaptive_vars(self):
        if self.binary_representation == 'adaptive_range':
            return np.array([self.rod.n_comp])
        else:
            return np.array([0])

    def get_adaptive_vars(self, nf_sol):
        if self.binary_representation == 'adaptive_range':
            return [nf_sol]
        else:
            return None

    def get_position_in_bit_array(self, i_group, i_var):
        start = i_var * self.n_qubits_per_var
        end = (i_var + 1) * self.n_qubits_per_var
        return start, end

    def get_real_number_object(self, i_group):
            return self.real_number

    def get_range_limits(self, i_group):
        assert(i_group == 0)
        return self.a_min, self.a_max

    def update_formulation(self, best_solution):
        self.update_nodal_force_polys()
        self.generate_cross_section_inverse_polys()

    def generate_nodal_force_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation in ['range', 'adaptive_range']:
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.ones(self.rod.n_comp)*lower_lim
            self.a_max = np.ones(self.rod.n_comp)*upper_lim
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_var, self.binary_representation, lower_lim, upper_lim)

        nf_polys = []
        for i_comp in range(self.rod.n_comp):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            nf_polys.append(self.real_number.evaluate(q))
            if i_comp == self.rod.n_comp-1:
                nf_polys.append(0.0)
        self.nf_polys = nf_polys

    def update_nodal_force_polys(self):
        self.initialize_discretization()
        nf_polys = []
        for i_comp in range(self.rod.n_comp):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            if self.binary_representation == 'adaptive_range':
                self.real_number.set_range(self.a_min[i_comp], self.a_max[i_comp])
            nf_polys.append(self.real_number.evaluate(q))
            if i_comp == self.rod.n_comp-1:
                nf_polys.append(0.0)
        self.nf_polys = nf_polys

    @abstractmethod
    def generate_cross_section_inverse_polys(self):
        pass

    def print_discretization(self):
        for i_comp in range(self.rod.n_comp):
            print('Component', i_comp)
            print('\tNodes', i_comp, i_comp+1)
            print('\t\tF'+str(i_comp)+' =', self.nf_polys[i_comp])
            print('\t\tF'+str(i_comp+1)+' =', self.nf_polys[i_comp+1])
            print('\tInverse of cross section area')
            print('\t\tA'+str(i_comp)+'_inv = ',self.cs_inv_polys[i_comp])

    def complementary_energy(self, nf, cs_inv):

        U = []
        for i_comp in range(self.rod.n_comp):
            a1 = nf[i_comp]
            a2 = nf[i_comp+1]
            U_comp = cs_inv[i_comp]*(self.rod.x[i_comp+1]-self.rod.x[i_comp])/(6.0*self.rod.E[i_comp])*(a1**2+a1*a2+a2**2)
            U.append(U_comp)
        # External Complementary Work.
        V = [0 for _ in range(self.rod.n_comp)]
        # Total Complementary Energy.
        PI = sum(U + V)
        return PI

    def generate_complementary_energy_poly(self):
        n_comp = self.rod.n_comp
        nf = self.nf_polys
        cs_inv = self.cs_inv_polys
        x = self.rod.x
        E = self.rod.E
        PI_poly = self.complementary_energy(nf, cs_inv)

        self.complementary_energy_poly = PI_poly

    def equilibrium_constraints(self, nf, cs_inv):

        eq_res = []
        for i_comp in range(self.rod.n_comp):
            a1 = nf[i_comp]
            a2 = nf[i_comp+1]

            div = (a2-a1)*cs_inv[i_comp]
            vol_force = (self.rod.x[i_comp+1]-self.rod.x[i_comp])*self.rod.rho[i_comp]*self.g
            eq = div + vol_force
            # eq_res.append(eq/self.rod.n_comp)
            eq_res.append(eq)
        return eq_res

    def traction_bc_constraints(self, nf):
        traction_bc = 0.0
        bc_res = nf[-1]-traction_bc
        return bc_res

    def constraints(self, nf, cs_inv):

        # Equilibrium.
        eq_cons = self.equilibrium_constraints(nf, cs_inv)
        eq_cons_squared = []
        for i_comp in range(self.rod.n_comp):
            eq_cons_squared.append(eq_cons[i_comp]**2)

        eq_cons_sq_sum = sum(eq_cons_squared)

        # Traction Boundary Condition.
        traction_bc_con = self.traction_bc_constraints(nf)
        traction_bc_con_sq =traction_bc_con**2

        return eq_cons, eq_cons_sq_sum, traction_bc_con_sq

    def generate_constraint_polys(self):

        nf = self.nf_polys
        cs_inv = self.cs_inv_polys
        eq_cons, eq_cons_sq_sum, traction_bc_con_sq = self.constraints(nf, cs_inv)

        # Only consider equilibrium constraint, since traction BC is built into ansatz.
        self.equilibrium_constraint_polys = eq_cons
        self.equilibrium_constraints_squared_sum_poly =  eq_cons_sq_sum

    def objective(self, complementary_energy, equilibrium_constraints_squared_sum, equilibrium_constraints):
        if self.constrained_opt_mode == 'penalty' or self.constrained_opt_mode == 'augmented_lagrangian':
            obj = complementary_energy + self.penalty_weight*equilibrium_constraints_squared_sum
            if self.constrained_opt_mode == 'augmented_lagrangian':
                for i, lagrange_multiplier in enumerate(self.lagrange_multipliers):
                    obj -= lagrange_multiplier * equilibrium_constraints[i]
            return obj
        else:
            raise Exception(f'Unknown mode ({self.constrained_opt_mode}) to compute objective.')

    def generate_problem_formulation(self, penalty_weight = 1.0, lagrange_multipliers = [], mode = 'penalty'):
        self.generate_complementary_energy_poly()
        self.generate_constraint_polys()

        self.constrained_opt_mode = mode
        self.penalty_weight = penalty_weight
        self.lagrange_multipliers = lagrange_multipliers

        self.poly = self.objective(
            self.complementary_energy_poly,
            self.equilibrium_constraints_squared_sum_poly,
            self.equilibrium_constraint_polys
        )

        self.binary_model = Model(self.poly)

        output = f'Number of binary variables: {len(self.binary_model.get_variables())}\n'
        self.print_and_log(output)

    def update_penalty_weight_in_problem_formulation(self, penalty_weight = 1.0):
        self.penalty_weight = penalty_weight
        self.poly = self.objective(
            self.complementary_energy_poly,
            self.equilibrium_constraints_squared_sum_poly,
            self.equilibrium_constraint_polys
        )

        self.binary_model = Model(self.poly)

    def plot_qubo_matrix_pattern(self, highlight_nodes=False, highlight_interactions=False):
        title = self.name + '\n QUBO Pattern \n'
        Q = self.get_qubo_matrix()
        binary_matrix = np.where(Q != 0, 1, 0)
        plt.figure()
        plt.suptitle(title)
        plt.imshow(binary_matrix,cmap='gray_r')

        if highlight_nodes:
            for i_node in range(self.rod.n_comp):
                x_pos = (i_node)*self.n_qubits_per_var - 0.5
                y_pos = x_pos
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    self.n_qubits_per_var,
                    self.n_qubits_per_var,
                    linewidth = 2,
                    edgecolor='red',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

        if highlight_interactions:
            for i_node in range(self.rod.n_comp-1):
                x_pos = (i_node)*self.n_qubits_per_var - 0.5
                y_pos = x_pos
                rect = patches.Rectangle(
                    (x_pos,y_pos),
                    2*self.n_qubits_per_var,
                    2*self.n_qubits_per_var,
                    linewidth = 2,
                    edgecolor='orange',
                    facecolor='none'
                )
                plt.gca().add_patch(rect)

    def visualize_qubo_matrix_pattern(self, show_fig=False, save_fig=False, save_tikz=False, suffix='', highlight_nodes=False, highlight_interactions=False):
        self.plot_qubo_matrix_pattern(highlight_nodes=highlight_nodes, highlight_interactions=highlight_interactions)
        if save_fig or save_tikz:
            assert(self.output_path is not None)
            file_name = os.path.join(self.output_path, self.name.lower().replace(' ', '_') + '_QUBO_pattern' + suffix)
            if highlight_nodes:
                suffix += '_highlight_nodes'
            if highlight_interactions:
                suffix += '_highlight_interactions'
            if save_fig:
                plt.savefig(file_name, dpi=600)
            if save_tikz:
                matplot2tikz.save(file_name + '.tex')
        if show_fig:
            plt.show()
        plt.close()

    def evaluate_result(self, result):

        # Decode solution, i.e., evaluate nodal forces and inverse of cross sections.
        nf_sol = self.decode_nodal_force_solution(result)
        cs_inv_sol = self.decode_cross_section_inverse_solution(result)
        # Compute complementary energy.
        PI_sol =  self.complementary_energy(nf_sol, cs_inv_sol)
        # Evaluate constraints.
        eq_cons, eq_cons_sq_sum, traction_bc_con_sq = self.constraints(nf_sol, cs_inv_sol)
        # Compute objective function.
        obj_sol = self.objective(PI_sol, eq_cons_sq_sum, eq_cons)

        solution = {
            'complementary_energy': PI_sol,
            'constraints': eq_cons,
            'constraints_squared_sum': eq_cons_sq_sum,
            'objective': obj_sol,
            'nf': nf_sol,
            'cs_inv': cs_inv_sol,
            'cs': [1/cs_inv for cs_inv in cs_inv_sol],
            'adaptive_vars': self.get_adaptive_vars(nf_sol)
        }
        return solution

    def get_best_solution(self, results=None):
        """
        Get best solution (minimum objective) from results computed or returned by a solver.

        :param results: Optional results to analyze. If not provided, will attempt to use `self.results` computed by
            a solver.

        :return: Best solution (dictionary).
        """
        if results is None and not hasattr(self, 'results'):
            raise Exception('Attempt to analyze results, but no results exist or have been passed.')
        elif results is None and hasattr(self, 'results'):
            results = self.results

        # Extract best solution (minimum objective value) from results.
        best_objective = np.inf
        for result in results:
            solution = self.evaluate_result(result)
            bit_array = self.get_bit_array(result)
            solution['bit_array'] = bit_array
            obj_sol = solution['objective']

            if obj_sol < best_objective:
                best_solution = solution
                best_objective = obj_sol

        # Prepare symbolic force and stress functions.
        nf_sol = best_solution['nf']
        cs_inv_sol = best_solution['cs_inv']
        force_sol, stress_sol = self.symbolic_force_and_stress_functions(nf_sol, cs_inv_sol)
        best_solution['force'] = force_sol
        best_solution['stress'] = stress_sol
        # Compute errors in force.
        error_l2_force_abs, error_l2_force_rel = self.rel_error_l2(self.force_analytic, force_sol)
        error_h1_force_abs, error_h1_force_rel = self.rel_error_h1(self.force_analytic, force_sol)
        best_solution['error_l2_abs'] = error_l2_force_abs
        best_solution['error_l2_rel'] = error_l2_force_rel
        best_solution['error_h1_abs'] = error_h1_force_abs
        best_solution['error_h1_rel'] = error_h1_force_rel

        output =  'Best solution (minimum objective):\n'
        output += '----------------------------------\n'
        output += f'Constraints (squared sum) {best_solution["constraints_squared_sum"]:.4e}\n'
        if hasattr(self, 'compare_designs'):
            output += self.compare_designs(best_solution)
        output += f'Force:\n\tRel. L2 error {error_l2_force_rel:.4e}\n\tRel. H1 error {error_h1_force_rel:.4e}\n'
        complementary_energy = best_solution['complementary_energy']
        complementary_energy_ref = self.PI_analytic
        rel_error_complementary_energy = abs(complementary_energy-complementary_energy_ref)/abs(complementary_energy_ref)
        output += f"Complementary energy = {complementary_energy}\n\tRel. error: {rel_error_complementary_energy:.4e}\n"
        self.print_and_log(output)

        return best_solution

    def decode_nodal_force_solution(self, result):
        nf_sol = []
        for nf_poly in self.nf_polys:
            if type(nf_poly) is Poly:
                if type(result) is SampleView:
                    nf_sol.append(self.decode_amplify_poly_with_bitstring(nf_poly,result._data))
                elif type(result) is SimpleNamespace:
                    nf_sol.append(self.decode_amplify_poly_with_bitstring(nf_poly,result.values))
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
            if type(cs_inv_poly) is Poly:
                if type(result) is SampleView:
                    cs_inv_sol.append(self.decode_amplify_poly_with_bitstring(cs_inv_poly,result._data))
                elif type(result) is SimpleNamespace:
                    cs_inv_sol.append(self.decode_amplify_poly_with_bitstring(cs_inv_poly,result.values))
                else:
                    cs_inv_sol.append(cs_inv_poly.decode(result.values))
            elif type(cs_inv_poly) in [float, np.float64]:
                cs_inv_sol.append(cs_inv_poly)
            else:
                print(type(cs_inv_poly))
                raise Exception('Unexpected type for cs_inv_poly')
        return cs_inv_sol

    def print_nodal_force_and_cross_section_inverse(self, nf_sol, cs_inv_sol):
        for i_comp in range(self.rod.n_comp):
            output = f'\tComponent {i_comp}\n'
            output+= f'\t\tNodes {i_comp}{i_comp+1}\n'
            output+= f'\t\t\tF{i_comp} = {nf_sol[i_comp]}\n'
            output+= f'\t\t\tF{i_comp+1} = {nf_sol[i_comp+1]}\n'
            output+= '\t\tCross section area\n'
            output+= f'\t\t\tA{i_comp} = {1/cs_inv_sol[i_comp]} ({self.A_analytic[i_comp]})\n'
            self.print_and_log(output)

    # Generate Symbolic Functions of Numerical Results.
    def symbolic_force_and_stress_functions(self, nf_sol, cs_inv_sol):
        # Force Function
        force_fun = []
        for i_comp in range(self.rod.n_comp):
            xi = self.rod.x[i_comp]
            xj = self.rod.x[i_comp+1]
            phi1, phi2 = self.basis(xi, xj, self.x_sym)
            force_fun.append(phi1*nf_sol[i_comp] + phi2*nf_sol[i_comp+1])

        # Stress Function
        stress_fun = []
        for i_comp in range(self.rod.n_comp):
            stress_fun.append(force_fun[i_comp]*cs_inv_sol[i_comp])
        return force_fun, stress_fun

    def show_error_over_objective(self):

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Objective')
        ax1.set_ylabel('Error')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.plot(self.objectives, self.errors_l2_rel, marker='*', color='tab:blue', linestyle='none', label='Error L2')
        ax1.plot(self.objectives, self.errors_h1_rel, marker='+', color='tab:orange', linestyle='none', label='Error H1')

        ax2 = ax1.twinx()
        ax2.plot(self.objectives, self.errors_comp_energy_rel, marker='o', color='tab:red', linestyle='none', label='Error Compl. Energy')
        ax2.set_yscale('log')
        ax2.set_ylabel('Error Complementary Energy')

        fig.show()

    # Plot Force Solutions.
    def plot_force(self, force_analyt, force_num, subtitle=None, file_name=None, save_fig=False, save_tikz=False):
        x_plot = []
        force_num_plot = []
        force_analyt_plot = []
        plt.figure()
        for i_node in range(self.rod.n_comp+1):
            plt.axvline(x=self.rod.x[i_node], color='gray', linestyle='--', linewidth=1.5)

        for i_comp in range(self.rod.n_comp):
            for i_x in np.linspace(self.rod.x[i_comp], self.rod.x[i_comp+1], 10):
                x_plot.append(i_x)
                force_num_plot.append(force_num[i_comp].subs(self.x_sym, i_x))
                force_analyt_plot.append(force_analyt[i_comp].subs(self.x_sym, i_x))

        plt.plot(x_plot, force_analyt_plot, 'k', label = "Analytical Solution")
        plt.plot(x_plot, force_num_plot, 'm', label = "Numerical Solution")

        for i_comp in range(self.rod.n_comp):
            plt.plot(self.rod.x[i_comp], force_num[i_comp].subs(self.x_sym, self.rod.x[i_comp]),'mo')

        plt.xlabel('x')
        plt.ylabel('Force')
        if subtitle:
            plt.title(self.name+'\n'+subtitle)
        else:
            plt.title(self.name)

        plt.legend()
        if save_fig:
            plt.savefig(file_name, dpi=600)
        if save_tikz:
            matplot2tikz.save(file_name+".tex")

    # Plot Stress Solutions.
    def plot_stress(self, stress_analyt, stress_num, subtitle=None, file_name=None, save_fig=False, save_tikz=False):
        x_plot = []
        stresses_num_plot = []
        stresses_analyt_plot = []
        plt.figure()
        for i_node in range(self.rod.n_comp+1):
            plt.axvline(x=self.rod.x[i_node], color='gray', linestyle='--', linewidth=1.5)

        for i_comp in range(self.rod.n_comp):
            for i_x in np.linspace(self.rod.x[i_comp], self.rod.x[i_comp+1], 10):
                x_plot.append(i_x)
                stresses_num_plot.append(stress_num[i_comp].subs(self.x_sym, i_x))
                stresses_analyt_plot.append(stress_analyt[i_comp].subs(self.x_sym, i_x))

        plt.plot(x_plot, stresses_analyt_plot, label = "Analytical Solution")
        plt.plot(x_plot, stresses_num_plot, label = "Numerical Solution")

        for i_comp in range(self.rod.n_comp):
            plt.plot(self.rod.x[i_comp], stress_num[i_comp].subs(self.x_sym, self.rod.x[i_comp]),'mo')

        plt.xlabel('x')
        plt.ylabel('Stress')
        if subtitle:
            plt.title(self.name+'\n'+subtitle)
        else:
            plt.title(self.name)

        plt.legend()
        if save_fig:
            plt.savefig(file_name, dpi=600)
        if save_tikz:
            matplot2tikz.save(file_name+".tex")

    # Relative Error betweeen Analytical and Numerical Force-Solution.
    def rel_error_l2(self, fun_analyt, fun_num):
        quad_norm_diff_fun = []
        quad_norm_fun = []
        for i_comp in range(self.rod.n_comp):
            diff_fun = fun_analyt[i_comp] - fun_num[i_comp]
            quad_norm_diff_fun.append(
                quad(
                    lambda x_int: (diff_fun.subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]
            )
            quad_norm_fun.append(
                quad(
                    lambda x_int: (fun_analyt[i_comp].subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]
            )
        error_abs = np.sqrt(sum(quad_norm_diff_fun))
        error_rel = error_abs / np.sqrt(sum(quad_norm_fun))

        return error_abs, error_rel

    # Relative Error betweeen Analytical and Numerical Force-Solution.
    def rel_error_h1(self, fun_analyt, fun_num):
        quad_norm_diff_fun = []
        quad_norm_fun = []
        for i_comp in range(self.rod.n_comp):
            diff_fun = fun_analyt[i_comp] - fun_num[i_comp]
            d_diff_fun_d_x = sp.diff(diff_fun)
            d_fun_analyt_d_x = sp.diff(fun_analyt[i_comp])
            quad_norm_diff_fun.append(
                quad(
                    lambda x_int: (diff_fun.subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]
                +
                quad(
                    lambda x_int: (d_diff_fun_d_x.subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]

            )
            quad_norm_fun.append(
                quad(
                    lambda x_int: (fun_analyt[i_comp].subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]
                +
                quad(
                    lambda x_int: (d_fun_analyt_d_x.subs(self.x_sym, x_int))**2,
                    self.rod.x[i_comp],
                    self.rod.x[i_comp+1]
                )[0]

            )
        error_abs = np.sqrt(sum(quad_norm_diff_fun))
        error_rel = error_abs / np.sqrt(sum(quad_norm_fun))

        return error_abs, error_rel
