from abc import ABC, abstractmethod
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
from prettytable import PrettyTable
from scipy.integrate import quad
import sympy as sp
import sys
import tikzplotlib

from engioptiqa.variables.real_number import RealNumber
from .rod_1d import Rod1D
from ..solution_emulator import SolutionEmulator

class BaseProblem(ABC):
    def __init__(self, rod, g, output_path=None):
        self.rod = rod
        self.g = g

        self.x_sym = sp.symbols('x')

        self.table = PrettyTable()
        self.table.field_names =\
            ['Cross Sections', 'Complementary Energy', 'Compliance']

        self.quad_method = None

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

    def set_output_path(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Folder '{output_path}' created successfully.")
        else:
            print(f"Folder '{output_path}' already exists.")
        self.log_file = os.path.join(output_path,'log.txt')
        self.output_path = output_path

        self.print_and_log(self.name+'\n')

    def analytical_complementary_energy_and_compliance(self, A_combi):

        n_comp = self.rod.n_comp
        x = self.rod.x
        E = self.rod.E
        rho = self.rod.rho
        g = self.g

        PI_combi = []
        C_combi = []
        for i_A_combi in range(len(A_combi)):

            A = A_combi[i_A_combi]
            tmp_rod_1d = Rod1D(n_comp, self.rod.L,A)

            # Stresses
            stress = self.compute_stress_function(tmp_rod_1d)

            # Displacement
            u = self.compute_displacement_function(stress, tmp_rod_1d)

            # Complementary Energy
            PI_elem = []
            for i_comp in range(n_comp):
                expr = A[i_comp]/E[i_comp] * stress[i_comp]**2
                PI_elem.append(1./2. * sp.integrate(expr,(self.x_sym, x[i_comp], x[i_comp+1])))
            PI_combi.append(sum(PI_elem))

            # Compliance
            C_elem = []
            for i_comp in range(n_comp):
                vol_force = rho[i_comp]*g
                expr = stress[i_comp]/E[i_comp]
                C_elem.append(A[i_comp]*sp.integrate(vol_force*u[i_comp], (self.x_sym, x[i_comp], x[i_comp+1])))
            C_combi.append(sum(C_elem))

            # Sanity check.
            assert(round(C_combi[-1], 5) == round(2*PI_combi[-1], 5))

        # Print as table.
        data = []
        for i in range(len(A_combi)):
            data.append({'Cross Sections': A_combi[i], \
                         'Complementary Energy': PI_combi[i], \
                         'Compliance': C_combi[i],})

        for row in data:
            self.table.add_row([row['Cross Sections'], row['Complementary Energy'], row['Compliance']])

        self.table.sortby = 'Complementary Energy'

        self.print_and_log(self.table.get_string()+'\n')

        self.PI_combi, self.C_combi, self.A_combi = PI_combi, C_combi, A_combi

    def print_and_log(self, output):
        print(output)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as file:
                file.write(output)

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

    # Generate Basis Functions.
    def basis(self, xi, xj, x_sym):
        phi1 = (xj - x_sym)/(xj-xi)
        phi2 = (x_sym-xi)/(xj-xi)
        return phi1, phi2

    def initialize_discretization(self):
        self.symbol_generator = SymbolGenerator(BinaryPoly)

    @abstractmethod
    def generate_discretization():
        pass

    def generate_nodal_force_polys(self, n_qubits_per_node, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.symbol_generator is not None)
        if binary_representation == 'range':
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.ones(self.rod.n_comp)*lower_lim
            self.a_max = np.ones(self.rod.n_comp)*upper_lim
        self.n_qubits_per_node = n_qubits_per_node
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_node, self.binary_representation)

        nf_polys = []
        for i_comp in range(self.rod.n_comp):
            q = self.symbol_generator.array(self.n_qubits_per_node)
            if binary_representation == 'range':
                nf_polys.append(self.real_number.evaluate(q, self.a_min[i_comp], self.a_max[i_comp]))
            else:
                nf_polys.append(self.real_number.evaluate(q))
            if i_comp == self.rod.n_comp-1:
                nf_polys.append(0.0)
        self.nf_polys = nf_polys

    def update_nodal_force_polys(self):
        self.initialize_discretization()
        nf_polys = []
        for i_comp in range(self.rod.n_comp):
            q = self.symbol_generator.array(self.n_qubits_per_node)
            if self.binary_representation == 'range':
                nf_polys.append(self.real_number.evaluate(q, self.a_min[i_comp], self.a_max[i_comp]))
            else:
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

    def constraints(self, nf, cs_inv):

        # Equilibrium.
        cons_eq = []
        for i_comp in range(self.rod.n_comp):
            a1 = nf[i_comp]
            a2 = nf[i_comp+1]

            div = (a2-a1)*cs_inv[i_comp]
            vol_force = (self.rod.x[i_comp+1]-self.rod.x[i_comp])*self.rod.rho[i_comp]*self.g
            eq = div + vol_force

            # Penalty term.
            # Manually.
            con_comp = eq**2
            cons_eq.append(con_comp)

        cons_eq = sum(cons_eq)/self.rod.n_comp

        # Traction Boundary Condition.
        traction_bc = 0.0
        cons_bc=(nf[-1]-traction_bc)**2

        return cons_eq, cons_bc

    def generate_constraint_polys(self):

        nf = self.nf_polys
        cs_inv = self.cs_inv_polys
        con_eq, con_bc = self.constraints(nf, cs_inv)

        # Only consider equilibrium constraint, since traction BC is built into ansatz.
        self.equilibrium_constraint_poly =  con_eq

    def generate_qubo_formulation(self, penalty_weight = 1.0):
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
                tikzplotlib.save(file_name + '.tex')
        if show_fig:
            plt.show()
        plt.close()

    def transform_to_dwave(self):

        coeff_dict = self.poly.asdict()
        constant = coeff_dict.get((), 0.0)
        linear = {k[0]: v for k, v in coeff_dict.items() if len(k) == 1}
        quadratic = {tuple(k): v for k, v in coeff_dict.items() if len(k) == 2}

        self.binary_quadratic_model_indices = BinaryQuadraticModelDWave(linear, quadratic, constant, vartype='BINARY')


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

    def analyze_results(self, results=None, analysis_plots=True, result_max=sys.maxsize):

        if results is None and not hasattr(self, 'results'):
            raise Exception('Attempt to analyze results, but no results exist or have been passed.')
        elif results is None and hasattr(self, 'results'):
            results = self.results

        self.errors_force_rel = [np.inf for _ in range(len(results))]
        solutions = [{'error_abs': np.inf, 'energy': np.inf} for _ in range(len(results))]
        solutions = [{} for _ in range(len(results))]

        self.errors_l2_rel = []
        self.errors_h1_rel = []
        self.objectives = []
        self.comp_energies = []
        self.errors_comp_energy_rel = []
        self.cs_inv =[]
        self.cs = []
        for i_result, result in enumerate(results):
            # Decode solution, i.e., evaluate nodal forces and inverse of cross sections.
            nf_sol = self.decode_nodal_force_solution(result)
            cs_inv_sol = self.decode_cross_section_inverse_solution(result)
            # Compute complementary energy.
            PI_sol =  self.complementary_energy(nf_sol, cs_inv_sol)
            # Evaluate constraints.
            con_eq_sol, con_bc_sol = self.constraints(nf_sol, cs_inv_sol)
            # Compute objective function.
            obj_sol = PI_sol + self.penalty_weight_equilibrium*con_eq_sol + 0.0*con_bc_sol
            # Compute symbolic force and stress functions.
            force_sol, stress_sol = self.symbolic_force_and_stress_functions(nf_sol, cs_inv_sol)
            # Compute error with respect to analytic solution.
            error_l2_force_abs, error_l2_force_rel = self.rel_error_l2(self.force_analytic, force_sol)
            error_h1_force_abs, error_h1_force_rel = self.rel_error_h1(self.force_analytic, force_sol)
            self.errors_force_rel[i_result] = error_l2_force_rel
            solutions[i_result]['force'] = force_sol
            solutions[i_result]['error_l2_abs'] = error_l2_force_abs
            solutions[i_result]['error_l2_rel'] = error_l2_force_rel
            solutions[i_result]['error_h1_abs'] = error_h1_force_abs
            solutions[i_result]['error_h1_rel'] = error_h1_force_rel
            solutions[i_result]['complementary_energy'] = PI_sol
            solutions[i_result]['constraints'] = con_eq_sol
            solutions[i_result]['objective'] = obj_sol
            solutions[i_result]['energy'] = self.get_energy(i_result)
            solutions[i_result]['frequency'] = self.get_frequency(i_result)
            solutions[i_result]['nf'] = nf_sol
            solutions[i_result]['cs_inv'] = cs_inv_sol

            self.errors_l2_rel.append(error_l2_force_rel)
            self.errors_h1_rel.append(error_h1_force_rel)
            self.objectives.append(obj_sol)
            self.comp_energies.append(PI_sol)
            self.errors_comp_energy_rel.append(abs(PI_sol-self.PI_analytic)/abs(self.PI_analytic))
            self.cs_inv.append(cs_inv_sol)
            self.cs.append([1/cs_inv for cs_inv in cs_inv_sol])

            # Output of analysis.
            if i_result < result_max:
                output = f'Solution {i_result}\n'
                output+= f"\tenergy = {solutions[i_result]['energy']}, frequency = {solutions[i_result]['frequency']}\n"
                self.print_and_log(output)
                self.print_nodal_force_and_cross_section_inverse(nf_sol, cs_inv_sol)
                self.print_solution_quantities(PI_sol, con_eq_sol, con_bc_sol, obj_sol, error_l2_force_abs, error_l2_force_rel)
                # Plot Solution
                if analysis_plots:
                    if self.output_path is not None:
                        file_name_force = os.path.join(self.output_path,'force_solution_'+str(i_result)+'.png')
                        file_name_stress = os.path.join(self.output_path,'stress_solution_'+str(i_result)+'.png')
                        file_name_rod = os.path.join(self.output_path,'rod_solution_'+str(i_result)+'.png')
                    else:
                        file_name_force = None
                        file_name_stress = None
                        file_name_rod = None
                    self.plot_force(self.force_analytic, force_sol, file_name=file_name_force, save_fig=self.save_fig)
                    self.plot_stress(self.stress_analytic, stress_sol, file_name=file_name_stress, save_fig=self.save_fig)
                    rod_tmp = Rod1D(self.rod.n_comp, self.rod.L, 0.0)
                    rod_tmp.set_cross_sections_from_inverse(cs_inv_sol)
                    rod_tmp.visualize(file_name=file_name_rod, save_fig=self.save_fig)

        output = 'Best solution (minimum objective):\n'
        i_min =np.argsort(self.objectives)
        i_sol = i_min[0]
        error_l2 = solutions[i_sol]['error_l2_rel']
        error_h1 = solutions[i_sol]['error_h1_rel']
        output += f'L2 Error {error_l2} {self.errors_l2_rel[i_sol]}\nH1 Error {error_h1} {self.errors_h1_rel[i_sol]}\n'
        self.print_and_log(output)

        return solutions

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

    def print_solution_quantities(self, PI_sol, con_eq_sol, con_bc_sol, obj_sol, error_force_abs, error_force_rel):
            output = f'\tComplementary Energy = {PI_sol:.15g} ({self.PI_analytic:.15g})\n'
            output+= f'\tConstraints = {con_eq_sol:.15g} {con_bc_sol:.15g}\n'
            con_eq_w_sol = self.penalty_weight_equilibrium*con_eq_sol
            output+= f'\tWeighted Constraints = {con_eq_w_sol:.15g} 0.0\n'
            output+= f'\tObjective = {obj_sol:.15g}\n'
            output+= f'\tAbsolute Error = {error_force_abs:.15g}\n'
            output+= f'\tRelative Error = {error_force_rel:.15g}\n'

            self.print_and_log(output)

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
            tikzplotlib.save(file_name+".tex")

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
            tikzplotlib.save(file_name+".tex")

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
