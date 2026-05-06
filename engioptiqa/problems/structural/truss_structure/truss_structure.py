from amplify import (
    Model,
    Poly,
    VariableGenerator,
)
from dimod.views.samples import SampleView
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys

from engioptiqa.problems import Problem
from engioptiqa.variables.real_number import RealNumber
from .truss_member import TrussMember

class TrussStructure(Problem):
    def __init__(self, output_path = None):
        """
        Class representing a truss structure analysis problem.

        :param output_path: Optional path for saving results.
        """

        super().__init__(output_path)
        self.nodes = {}  # Dictionary to store nodes: {node_id: (x, y)}
        self.n_nodes = 0
        self.members = []  # List to store truss members
        self.n_members = 0
        self.loads = {}  # Dictionary to store external forces: {node_id: (Fx, Fy)}
        self.supports = {}  # Dictionary to store support conditions: {node_id: (x_fixed, y_fixed)}

        self.penalty_weight = 0.0

    def capabilities(self):
        return super().capabilities() | {"outeropt_penalty"}

    def add_node(self, node_id, coordinates):
        """
        Add a node to the truss structure.

        :param node_id: Unique identifier for the node (e.g., integer or string).
        :param coordinates: Tuple (x, y) representing the node's position.
        """
        self.nodes[node_id] = coordinates
        self.n_nodes += 1

    def add_member(self, node_0_id, node_1_id, A=None, E=None, member_id=None):
        """
        Add a truss member to the structure.

        :param node_0_id: ID of the first node.
        :param node_1_id: ID of the second node.
        :param A: Cross-sectional area of the member (optional).
        :param E: Young's modulus of the member (optional).
        :param member_id: Unique identifier for the member (optional).
        """
        if node_0_id not in self.nodes or node_1_id not in self.nodes:
            raise ValueError("Both nodes must exist in the structure before adding a member.")

        node_0 = self.nodes[node_0_id]
        node_1 = self.nodes[node_1_id]
        member = TrussMember(node_0_id, node_1_id, node_0, node_1, A, E, member_id)
        self.members.append(member)
        self.n_members += 1

    def set_member_areas(self, member_areas):
        """
        Set the cross-sectional areas for all members.

        :param member_areas: List of areas corresponding to each member.
        """
        if len(member_areas) != self.n_members:
            raise ValueError("Length of areas list must match the number of members.")

        for i, member in enumerate(self.members):
            member.A = member_areas[i]

    def get_member_areas(self):
        """
        Get the cross-sectional areas for all members.
        """
        return [member.A for member in self.members]

    def get_member_info(self):
        """
        Retrieve information about all members in the structure.

        :return: List of dictionaries containing member properties.
        """
        member_info = []
        for i, member in enumerate(self.members):
            print()
            info = {
                "Member": i,
                "Node 0": member.node_id_0,
                "Node 1": member.node_id_1,
                "Length": member.length,
                "Direction Cosines (Node 0)": member.direction_cosines_0,
                "Direction Cosines (Node 1)": member.direction_cosines_1,
                "Area (A)": member.A,
                "Young's Modulus (E)": member.E,
            }
            member_info.append(info)
        return member_info

    def get_node_info(self):
        """
        Retrieve information about all nodes in the structure.

        :return: Dictionary of node IDs and their coordinates.
        """
        return self.nodes

    def add_load(self, node_id, force):
        """
        Add an external load to a node.

        :param node_id: ID of the node where the load is applied.
        :param force: Tuple (Fx, Fy) representing the force components in x and y directions.
        """
        if node_id not in self.nodes:
            raise ValueError("Node must exist in the structure before adding a load.")

        self.loads[node_id] = force

    def get_load_info(self):
        """
        Retrieve information about all loads in the structure.

        :return: Dictionary of node IDs and their applied forces.
        """
        return self.loads

    def add_support(self, node_id, x_fixed=True, y_fixed=True):
        """
        Add a support condition to a node.

        :param node_id: ID of the node where the support is applied.
        :param x_fixed: Boolean indicating if the x-direction is fixed (default: True).
        :param y_fixed: Boolean indicating if the y-direction is fixed (default: True).
        """
        if node_id not in self.nodes:
            raise ValueError("Node must exist in the structure before adding a support.")

        self.supports[node_id] = (x_fixed, y_fixed)

    def get_support_info(self):
        """
        Retrieve information about all supports in the structure.

        :return: Dictionary of node IDs and their support conditions.
        """
        return self.supports

    def visualize(self, subtitle=''):
        """
        Visualize the truss structure, including nodes, members, loads, and supports.

        :param subtitle: Subtitle for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot nodes
        for node_id, (x, y) in self.nodes.items():
            ax.plot(x, y, 'o', color='green', zorder=2)  # Nodes as black circles
            ax.text(x - 0.075, y - 0.075, f"{node_id}", fontsize=12, zorder=3)

        # Plot members
        A_max = max([member.A for member in self.members if member.A is not None]) if any(member.A > 0. for member in self.members) else 1.0
        # print(f'Max Area: {A_max}')

        for member in self.members:
            x0, y0 = member.get_coords(local_node_id = 0)
            x1, y1 = member.get_coords(local_node_id = 1)
            lw = member.A / A_max * 5 if member.A is not None else 1.0
            ax.plot([x0, x1], [y0, y1], color='gray',lw=lw, label="Member", zorder=1)  # Members as black lines

        # Plot loads
        for node_id, (Fx, Fy) in self.loads.items():
            x, y = self.nodes[node_id]
            F_norm = (Fx**2 + Fy**2)**0.5
            ax.arrow(x, y, Fx / (2*F_norm), Fy / (2*F_norm), color='red', zorder=1,
                     head_width=0.025, length_includes_head=True)  # Loads as red arrows

        # Plot supports
        for node_id, (x_fixed, y_fixed) in self.supports.items():
            x, y = self.nodes[node_id]
            if x_fixed and y_fixed:
                ax.plot(x, y, 's', color='blue', zorder=2)  # Fixed supports as blue squares
            elif x_fixed:
                ax.plot(x, y, '>', color='blue', zorder=2)  # Pinned supports (x) as blue triangles
            elif y_fixed:
                ax.plot(x, y, '^', color='blue', zorder=2)  # Pinned supports (y) as blue triangles

        # Custom legend entries
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Members'),
            Line2D([0], [0], marker='o', color='green', markersize=8, label='Nodes', linestyle='None'),
            Line2D([0], [0], marker='s', color='blue', markersize=8, label='Supports', linestyle='None'),
            Line2D([0], [0], color='red', lw=2, label='Loads (Scaled)'),
        ]

        # Add legend
        ax.legend(handles=legend_elements, loc="upper right")

        # Set plot properties
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_axisbelow(True)
        ax.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Truss Structure: " + subtitle)
        if self.output_path is not None:
            plt.savefig(self.output_path / f"truss_structure_{subtitle.lower().replace(' ', '_')}.png", dpi=600)
        else:
            plt.show()

    def compute_member_forces(self):
        """
        Compute axial forces in the truss members using the method of joints.
        :return: Dictionary of member forces: {(node_1_id, node_2_id): force}.
        """
        # Number of nodes
        num_nodes = len(self.nodes)

        # Initialize global force matrix and displacement vector
        num_equations = 2 * num_nodes  # Two equations per node (Fx and Fy)
        global_matrix = np.zeros((num_equations, num_equations))
        global_force = np.zeros(num_equations)

        # Map node IDs to equation indices
        node_index_map = {node_id: i for i, node_id in enumerate(self.nodes.keys())}

        # Assemble global force vector
        for node_id, (Fx, Fy) in self.loads.items():
            index_x = 2 * node_index_map[node_id]
            index_y = index_x + 1
            global_force[index_x] += Fx
            global_force[index_y] += Fy

        # Assemble global stiffness matrix
        for member in self.members:
            node_id_0 = member.node_id_0
            node_id_1 = member.node_id_1
            # Get direction cosines
            l, m = member.direction_cosines_0
            L = member.length

            # Get indices for the nodes
            index_0_x = 2 * node_index_map[node_id_0]
            index_0_y = index_0_x + 1
            index_1_x = 2 * node_index_map[node_id_1]
            index_1_y = index_1_x + 1

            # Stiffness matrix for the member
            k = member.E * member.A / L
            member_matrix = k * np.array([
                [l**2, l*m, -l**2, -l*m],
                [l*m, m**2, -l*m, -m**2],
                [-l**2, -l*m, l**2, l*m],
                [-l*m, -m**2, l*m, m**2]
            ])

            # Add member stiffness matrix to global stiffness matrix
            global_matrix[index_0_x:index_0_y+1, index_0_x:index_0_y+1] += member_matrix[:2, :2]
            global_matrix[index_0_x:index_0_y+1, index_1_x:index_1_y+1] += member_matrix[:2, 2:]
            global_matrix[index_1_x:index_1_y+1, index_0_x:index_0_y+1] += member_matrix[2:, :2]
            global_matrix[index_1_x:index_1_y+1, index_1_x:index_1_y+1] += member_matrix[2:, 2:]

        # Apply support conditions
        for node_id, (x_fixed, y_fixed) in self.supports.items():
            index_x = 2 * node_index_map[node_id]
            index_y = index_x + 1
            if x_fixed:
                global_matrix[index_x, :] = 0
                global_matrix[:, index_x] = 0
                global_matrix[index_x, index_x] = 1
                global_force[index_x] = 0
            if y_fixed:
                global_matrix[index_y, :] = 0
                global_matrix[:, index_y] = 0
                global_matrix[index_y, index_y] = 1
                global_force[index_y] = 0

        # Solve for displacements
        displacements = np.linalg.solve(global_matrix, global_force)
        compliance = np.inner(global_force, displacements)

        # Compute member forces
        member_forces = {}
        for i_member, member in enumerate(self.members):
            node_id_0 = member.node_id_0
            node_id_1 = member.node_id_1
            l, m = member.direction_cosines_0
            L = member.length
            index_0_x = 2 * node_index_map[node_id_0]
            index_0_y = index_0_x + 1
            index_1_x = 2 * node_index_map[node_id_1]
            index_1_y = index_1_x + 1

            # Relative displacements
            u1 = displacements[index_0_x]
            v1 = displacements[index_0_y]
            u2 = displacements[index_1_x]
            v2 = displacements[index_1_y]

            # Axial force in the member
            axial_force = member.E * member.A / L * (l * (u2 - u1) + m * (v2 - v1))
            member_forces[i_member] = axial_force

        return member_forces, compliance

    def initialize_discretization(self):
        self.variable_generator = VariableGenerator()

    def generate_discretization(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        self.initialize_discretization()
        self.generate_member_stress_polys(n_qubits_per_var, binary_representation, lower_lim, upper_lim)
        self.generate_member_area_polys()

    def get_number_of_continuous_vars(self):
        return self.n_members

    def update_formulation(self):
        self.update_member_stress_polys()
        self.generate_member_area_polys()

    def generate_member_stress_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation == 'range':
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.ones(len(self.members))*lower_lim
            self.a_max = np.ones(len(self.members))*upper_lim
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_var, self.binary_representation, lower_lim, upper_lim)

        member_stress_polys = []
        for _ in self.members:
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            member_stress_polys.append(self.real_number.evaluate(q))
        self.member_stress_polys = member_stress_polys

    def update_member_stress_polys(self):
        self.initialize_discretization()
        member_stress_polys = []
        for i_member, _ in enumerate(self.members):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            if self.binary_representation == 'range':
                self.real_number.set_range(self.a_min[i_member], self.a_max[i_member])
            member_stress_polys.append(self.real_number.evaluate(q))
        self.member_stress_polys = member_stress_polys

    def generate_member_area_polys(self):
        member_area_polys = []
        for member in self.members:
            member_area_polys.append(member.A)
        self.member_area_polys = member_area_polys

    def complementary_energy(self, member_stresses, member_areas):
        U = []
        for i_member, member in enumerate(self.members):
            A = member_areas[i_member]
            L = member.length
            E = member.E
            stress = member_stresses[i_member]
            U.append(stress**2* A * L / (2*E))
        V = [0 for _ in range(len(self.members))]
        # Total Complementary Energy.
        PI = sum(U + V)
        return PI

    def generate_complementary_energy_poly(self):
        member_stresses = self.member_stress_polys
        member_areas = self.member_area_polys
        PI_poly = self.complementary_energy(member_stresses, member_areas)
        self.complementary_energy_poly = PI_poly

    def total_volume(self, member_areas):
        total_volume = 0.0
        for i_member, member in enumerate(self.members):
            L = member.length
            A = member_areas[i_member]
            total_volume += A * L
        return total_volume

    def joint_residuals_squared(self, member_stresses, member_areas):

        # Equilibrium (automatically fulfilled without body forces)

        # Traction boundary conditions (joint forces must be zero)
        joint_forces_x = [0. for _ in range(len(self.nodes))]
        joint_forces_y = [0. for _ in range(len(self.nodes))]

        # Add load contributions
        for node_id, (Fx, Fy) in self.loads.items():
            joint_forces_x[node_id] += Fx
            joint_forces_y[node_id] += Fy

        # Add member contributions
        for i_member, member in enumerate(self.members):
            node_id_0 = member.node_id_0
            node_id_1 = member.node_id_1

            F = member_stresses[i_member] * member_areas[i_member]

            # Get direction cosines
            l, m = member.direction_cosines_0
            joint_forces_x[node_id_0] += F * l
            joint_forces_y[node_id_0] += F * m

            l, m = member.direction_cosines_1
            joint_forces_x[node_id_1] += F * l
            joint_forces_y[node_id_1] += F * m

        # Sum the squared residual forces over all joints (ignore supports)
        cons_bc = 0.0
        n_loads = len(self.loads)
        if n_loads > 0:
            total_mag = 0.0
            for x, y in self.loads.values():
                total_mag += np.sqrt(x**2+ y**2)
            scale = total_mag/n_loads
        else:
            raise Exception('No loads specified.')

        for i_node in range(self.n_nodes):
            x_fixed = y_fixed = False
            if i_node in self.supports.keys():
                x_fixed, y_fixed = self.supports[i_node]
            if not x_fixed:
                cons_bc += (joint_forces_x[i_node]/scale)**2
            if not y_fixed:
                cons_bc += (joint_forces_y[i_node]/scale)**2

        return cons_bc

    def generate_joint_residuals_poly(self):
        member_stresses = self.member_stress_polys
        member_areas = self.member_area_polys
        self.joint_residuals_poly = self.joint_residuals_squared(member_stresses, member_areas)

    def generate_problem_formulation(self, penalty_weight):
        self.generate_complementary_energy_poly()
        self.generate_joint_residuals_poly()
        self.penalty_weight = penalty_weight
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight * self.joint_residuals_poly
        self.binary_model = Model(self.poly)

    def analyze_results(self, results=None, analysis_plots=True, compute_errors=True, result_max=sys.maxsize):

        if results is None and not hasattr(self, 'results'):
            raise Exception('Attempt to analyze results, but no results exist or have been passed.')
        elif results is None and hasattr(self, 'results'):
            results = self.results

        solutions = [{'objective': np.inf} for _ in range(len(results))]
        best_solution = None
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)
            member_stresses_sol = self.decode_member_stress_solution(result)
            member_areas_sol = self.decode_member_area_solution(result)
            member_forces_sol = [member_stresses_sol[i]*member_areas_sol[i] for i in range(len(member_stresses_sol))]
            complementary_energy_sol = self.complementary_energy(member_stresses_sol, member_areas_sol)
            volume_sol = self.total_volume(member_areas_sol)
            joint_residuals_squared_sol = self.joint_residuals_squared(member_stresses_sol, member_areas_sol)
            constraints_sol = joint_residuals_squared_sol
            if hasattr(self, 'target_volume'):
                constraints_sol += (volume_sol-self.target_volume)**2
            objective_sol = complementary_energy_sol + self.penalty_weight * constraints_sol

            if best_solution is None or objective_sol < best_solution['objective']:
                best_solution = solutions[i_result]

            solutions[i_result]['bit_array'] = bit_array
            solutions[i_result]['member_forces'] = member_forces_sol
            solutions[i_result]['continuous_vars'] = member_stresses_sol
            solutions[i_result]['member_areas'] = member_areas_sol
            solutions[i_result]['complementary_energy'] = complementary_energy_sol
            solutions[i_result]['volume'] = volume_sol
            solutions[i_result]['joint_residuals_squared'] = joint_residuals_squared_sol
            solutions[i_result]['constraints'] = constraints_sol
            solutions[i_result]['objective'] = objective_sol

            if hasattr(self, 'ts_ref'):
                rel_error_forces, area_mismatch, rel_error_compliance = self.compare_with_reference_solution(solutions[i_result])
                solutions[i_result]['avg_rel_error_forces'] = np.average(rel_error_forces)
                solutions[i_result]['rel_error_compliance'] = rel_error_compliance
                solutions[i_result]['areas_matching'] = not any(area_mismatch)


        print('Best solution (minimum objective):')
        print(f"Objective: {best_solution['objective']}")
        print(f"Complementary Energy: {best_solution['complementary_energy']}")
        print(f"Volume: {best_solution['volume']}")
        print(f"Joint Residuals (squared): {best_solution['joint_residuals_squared']}")
        print(f"Member Forces: {best_solution['member_forces']}")
        print(f"Member Stresses: {best_solution['continuous_vars']}")
        print(f"Member Areas: {best_solution['member_areas']}")
        if hasattr(self, 'ts_ref'):
            print(f"Average Relative Error in Member Forces: {best_solution['avg_rel_error_forces']}")
            print(f"Relative Error in Compliance: {best_solution['rel_error_compliance']}")
            print(f"Areas Matching Reference Solution: {best_solution['areas_matching']}")

        return solutions

    def decode_member_stress_solution(self, result):
        member_stress_sol = []
        for member_stress_poly in self.member_stress_polys:
            if type(result) is SampleView:
                member_stress_sol.append(self.decode_amplify_poly_with_bitstring(member_stress_poly,result._data))
            else:
                member_stress_sol.append(member_stress_poly.decode(result.values))
        return member_stress_sol

    def decode_member_area_solution(self, result):
        member_area_sol = []
        for member_area_poly in self.member_area_polys:
            if isinstance(member_area_poly, Poly):
                if type(result) is SampleView:
                    member_area_sol.append(self.decode_amplify_poly_with_bitstring(member_area_poly,result._data))
                else:
                    member_area_sol.append(member_area_poly.decode(result.values))
            else:
                member_area_sol.append(member_area_poly)
        return member_area_sol

    def set_reference_solution(self, ts_ref):
        self.ts_ref = ts_ref

    def compare_with_reference_solution(self, solution):
        if not hasattr(self, 'ts_ref'):
            raise Exception('No reference solution set for comparison')

        # Ensure that nodes in the reference solution are matching
        for node_id, node_coords in self.ts_ref.nodes.items():
            if self.nodes[node_id] != node_coords:
                raise Exception('Nodes in the provided reference solution are not matching.')

        # Compute reference solution
        self.member_forces_ref_sol, self.compliance_ref_sol = self.ts_ref.compute_member_forces()

        # Compute relative error in member forces and compliance
        member_forces = solution['member_forces']
        rel_error_forces = []
        for i_member in range(len(self.member_forces_ref_sol)):
            rel_error_force = abs((self.member_forces_ref_sol[i_member]-member_forces[i_member])) \
                /abs(self.member_forces_ref_sol[i_member])
            rel_error_forces.append(rel_error_force)

        compliance = 2.*solution['complementary_energy']
        rel_error_compliance = abs((self.compliance_ref_sol-compliance))/abs(self.compliance_ref_sol)


        # Check if members present in solution and reference solution match
        ref_members = {
            (member.node_id_0, member.node_id_1, member.A)
            for member in self.ts_ref.members
        }

        # Check if members with non-zero cross-sectional area (A > 0) also exist in reference solution
        area_mismatch = [False for _ in self.members]
        for i_member, member in enumerate(self.members):
            member_area = solution['member_areas'][i_member]
            if member_area > 0.:
                key = (member.node_id_0, member.node_id_1, member_area)
                if key not in ref_members:
                    area_mismatch[i_member] = True

        # Detect members that exist in the reference solution but are missing in the solution
        member_areas = solution['member_areas']
        members = {
            (member.node_id_0, member.node_id_1, member_areas[i_member])
            for i_member, member in enumerate(self.members)
        }
        missing_in_ts = ref_members - members
        for i_member, key in enumerate(ref_members):
            if key in missing_in_ts:
                area_mismatch[i_member] = True



        return rel_error_forces, area_mismatch, rel_error_compliance