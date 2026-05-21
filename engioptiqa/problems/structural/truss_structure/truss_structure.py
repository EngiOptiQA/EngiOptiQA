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
from types import SimpleNamespace

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
        self.nsd = 2
        self.nodes = {}  # Dictionary to store nodes: {node_id: (x, y)}
        self.n_nodes = 0
        self.members = []  # List to store truss members
        self.n_members = 0
        self.optional_members = []  # List to store optional members (if any)
        self.n_optional_members = 0
        self.loads = {}  # Dictionary to store external forces: {node_id: (Fx, Fy)}
        self.supports = {}  # Dictionary to store support conditions: {node_id: (x_fixed, y_fixed)}

        self.penalty_weight = 0.0

    def capabilities(self):
        return super().capabilities() | {"outeropt_penalty", "outeropt_augmented_lagrangian"}

    def add_node(self, node_id, coordinates):
        """
        Add a node to the truss structure.

        :param node_id: Unique identifier for the node (e.g., integer or string).
        :param coordinates: Tuple (x, y) representing the node's position.
        """
        if len(coordinates) != self.nsd:
            raise Exception(f"Only {self.nsd}D coordinates allowed!")
        self.nodes[node_id] = coordinates
        self.n_nodes += 1

    def add_member(self, node_0_id, node_1_id, A=None, E=None, member_id=None, optional=True):
        """
        Add a truss member to the structure.

        :param node_0_id: ID of the first node.
        :param node_1_id: ID of the second node.
        :param A: Cross-sectional area of the member (optional).
        :param E: Young's modulus of the member (optional).
        :param member_id: Unique identifier for the member (optional).
        :param optional: Boolean indicating if the member is optional (default: True).
        """
        if node_0_id not in self.nodes or node_1_id not in self.nodes:
            raise ValueError("Both nodes must exist in the structure before adding a member.")

        node_0 = self.nodes[node_0_id]
        node_1 = self.nodes[node_1_id]
        member = TrussMember(node_0_id, node_1_id, node_0, node_1, A, E, member_id)
        self.members.append(member)
        self.n_members += 1
        if optional:
            self.optional_members.append(member)
            self.n_optional_members += 1

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

    def get_n_fixed(self):
        n_fixed = 0
        for i_node in range(self.n_nodes):
            x_fixed = y_fixed = False
            if i_node in self.supports.keys():
                x_fixed, y_fixed = self.supports[i_node]
            if x_fixed:
                n_fixed += 1
            if y_fixed:
                n_fixed += 1
        return n_fixed

    def check_statically_determinate(self):
        """
        Check if the truss is statically determinate.

        :return: Dictionary with information about the truss's determinacy.
        """

        # Number of (existent) members
        m = 0
        for member in self.members:
            if member.A>0.:
                m += 1
        j = len(self.nodes) # Number of joints (nodes)
        # Reaction count: sum of fixed directions over supports
        r = 0
        for _, (xf, yf) in self.supports.items():
            if xf:
                r += 1
            if yf:
                r += 1

        # Determinacy condition for truss: m + r = nsd*j
        degree = (m + r) - self.nsd * j
        if degree == 0:
            condition = 'determinate'
        elif degree > 0:
            condition = 'indeterminate'
        else:  # degree < 0
            condition = 'unstable'

        return {
            'm': m,
            'j': j,
            'r': r,
            'condition': condition,
            'degree': degree,
        }

    def get_number_of_existent_members(self):
        n_existent_members = 0
        for member in self.members:
            if member.exists:
                n_existent_members += 1
        return n_existent_members

    def visualize(self, subtitle='', interactive=False):
        """
        Visualize the truss structure, including nodes, members, loads, and supports.

        :param subtitle: Subtitle for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        x_max = max([coord[0] for coord in self.nodes.values()])
        x_min = min([coord[0] for coord in self.nodes.values()])
        dx = x_max - x_min
        y_max = max([coord[1] for coord in self.nodes.values()])
        y_min = min([coord[1] for coord in self.nodes.values()])
        dy = y_max - y_min
        dx_dy_mean = np.mean([dx,dy])

        # Plot nodes
        for node_id, (x, y) in self.nodes.items():
            if node_id not in self.supports.keys():
                ax.plot(x, y, 'o', color='green', zorder=2)
            ax.text(x - 0.05*dx_dy_mean, y - 0.05*dx_dy_mean, f"{node_id}", fontsize=12, zorder=3)

        # Plot members
        A_max = max([member.A for member in self.members if member.exists]) if any(member.A > 0. for member in self.members) else 1.0

        for i_member, member in enumerate(self.members):
            if member.exists:
                x0, y0 = member.get_coords(local_node_id = 0)
                x1, y1 = member.get_coords(local_node_id = 1)
                lw = max(member.A / A_max * 5, 0.1)
                label = "Member" if i_member == 0 else None
                if member in self.optional_members:
                    ax.plot([x0, x1], [y0, y1], color='gray', linestyle='dashed', lw=lw, label=label, zorder=1)
                else:
                    ax.plot([x0, x1], [y0, y1], color='gray',lw=lw, label=label, zorder=1)

        # Plot loads
        for node_id, (Fx, Fy) in self.loads.items():
            x, y = self.nodes[node_id]
            F_norm = (Fx**2 + Fy**2)**0.5
            ax.arrow(x, y, Fx / (2*F_norm) * dx, Fy / (2*F_norm) * dy, color='red', zorder=1,
                     head_width=0.02*np.sqrt(dx**2 + dy**2), length_includes_head=True)  # Loads as red arrows

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
        if interactive:
            plt.show()
        plt.close(fig)

    def get_existent_members_and_involved_nodes(self):
        """
        Filter out non-existent members (with zero cross-sectional area) and nodes without any existent member.
        """

        existent_members = {}
        involved_nodes = []

        for i_member, member in enumerate(self.members):
            if member.A > 0.:
                existent_members[i_member] = member
                node_id_0 = member.node_id_0
                node_id_1 = member.node_id_1
                involved_nodes.append(node_id_0)
                involved_nodes.append(node_id_1)
        involved_nodes = list(set(involved_nodes))

        return existent_members, involved_nodes



    def compute_member_forces(self):
        """
        Compute axial forces in the truss members using the method of joints.
        :return: Dictionary of member forces: {(node_1_id, node_2_id): force}.
        """

        members, nodes = self.get_existent_members_and_involved_nodes()

        # Number of nodes
        num_nodes = len(nodes)

        # Initialize global force matrix and displacement vector
        num_equations = 2 * num_nodes  # Two equations per node (Fx and Fy)
        global_matrix = np.zeros((num_equations, num_equations))
        global_force = np.zeros(num_equations)

        # Map node IDs to equation indices
        node_index_map = {node_id: i for i, node_id in enumerate(nodes)}

        # Assemble global force vector
        for node_id, (Fx, Fy) in self.loads.items():
            index_x = 2 * node_index_map[node_id]
            index_y = index_x + 1
            global_force[index_x] += Fx
            global_force[index_y] += Fy

        # Assemble global stiffness matrix
        for member in members.values():
            if member.A > 0.:
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
            if node_id in nodes:
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
        for i_member, member in members.items():
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

    def has_adaptive_variables(self):
        return self.binary_representation == 'adaptive_range'

    def get_number_of_adaptive_vars(self):
        if self.binary_representation == 'adaptive_range':
            return np.array([self.n_members])
        else:
            return np.array([0])

    def get_adaptive_vars(self, member_stresses_sol, member_areas_sol):
        if self.binary_representation == 'adaptive_range':
            return [member_stresses_sol]
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
        if self.binary_representation == 'adaptive_range':
            return self.a_min, self.a_max
        else:
            return None, None

    def update_formulation(self, best_solution=None):
        self.update_member_stress_polys()
        self.generate_member_area_polys()

    def generate_member_stress_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation in ['range', 'adaptive_range']:
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.ones(len(self.members))*lower_lim
            self.a_max = np.ones(len(self.members))*upper_lim
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_var, self.binary_representation, lower_lim, upper_lim)

        member_stress_polys = []
        for i_member, _ in enumerate(self.members):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var, name=f"q_S_{i_member}")
            member_stress_polys.append(self.real_number.evaluate(q))
        self.member_stress_polys = member_stress_polys

    def update_member_stress_polys(self):
        self.initialize_discretization()
        member_stress_polys = []
        for i_member, _ in enumerate(self.members):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var, name=f"q_S_{i_member}")
            if self.binary_representation == 'adaptive_range':
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

    def joint_residuals(self, member_stresses, member_areas):

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
        n_loads = len(self.loads)
        if n_loads > 0:
            total_mag = 0.0
            for x, y in self.loads.values():
                total_mag += np.sqrt(x**2+ y**2)
            scale = total_mag/n_loads
        else:
            raise Exception('No loads specified.')

        bc_cons_x = []
        bc_cons_y = []

        for i_node in range(self.n_nodes):
            x_fixed = y_fixed = False
            if i_node in self.supports.keys():
                x_fixed, y_fixed = self.supports[i_node]
            if not x_fixed:
                bc_cons_x.append(joint_forces_x[i_node]/scale)
            if not y_fixed:
                bc_cons_y.append(joint_forces_y[i_node]/scale)

        return bc_cons_x, bc_cons_y

    def joint_residuals_squared_sum(self, member_stresses, member_areas):

        bc_cons_x, bc_cons_y = self.joint_residuals(member_stresses, member_areas)

        bc_cons_poly = 0.0
        for bc_con in bc_cons_x:
            bc_cons_poly += bc_con**2
        for bc_con in bc_cons_y:
            bc_cons_poly += bc_con**2
        return bc_cons_x, bc_cons_y, bc_cons_poly

    def generate_joint_residuals_poly(self):
        member_stresses = self.member_stress_polys
        member_areas = self.member_area_polys

        bc_cons_x, bc_cons_y, bc_cons_sq_sum = self.joint_residuals_squared_sum(member_stresses, member_areas)
        self.joint_residuals_squared_sum_poly = bc_cons_sq_sum
        self.joint_residual_polys = bc_cons_x + bc_cons_y

    def generate_constraint_polys(self):
        self.generate_joint_residuals_poly()
        self.constraint_polys = self.joint_residual_polys
        self.constraints_sum_squared_poly = self.joint_residuals_squared_sum_poly

    def get_n_constraints(self):
        n_free = self.nsd*self.n_nodes - self.get_n_fixed()
        return n_free

    def generate_problem_formulation(self, penalty_weight=1.0, lagrange_multipliers=[], mode='penalty'):
        self.generate_complementary_energy_poly()
        self.generate_constraint_polys()

        self.constrained_opt_mode = mode
        self.penalty_weight = penalty_weight
        self.lagrange_multipliers = lagrange_multipliers

        self.poly = self.objective(
            self.complementary_energy_poly,
            self.constraints_sum_squared_poly,
            self.constraint_polys
        )

        # self.generate_objective_poly(penalty_weight=penalty_weight, lagrange_multipliers=lagrange_multipliers, mode=mode)
        self.binary_model = Model(self.poly)

    def objective(self, complementary_energy, constraints_squared_sum, constraints):
        if  self.constrained_opt_mode == 'penalty' or  self.constrained_opt_mode == 'augmented_lagrangian':
            obj = complementary_energy + self.penalty_weight * constraints_squared_sum
            if self.constrained_opt_mode == 'augmented_lagrangian':
                n_constraints = self.get_n_constraints()
                if len(self.lagrange_multipliers) != (n_constraints):
                    raise Exception('Number of Lagrange multipliers must be equal to number of constraints' \
                                    f'({n_constraints}).')
                for i, lagrange_multiplier in enumerate(self.lagrange_multipliers):
                    obj -= lagrange_multiplier * constraints[i]
            return obj
        else:
            raise Exception(f'Unknown mode ({self.constrained_opt_mode}) to compute objective.')

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

        # Prepare mapping from variable IDs to bitstring positions for decoding.
        self.bitstring_pos = {}
        i_pos = 0
        for var in self.binary_model.variables:
            self.bitstring_pos[var.id] = i_pos
            i_pos +=1

        best_solution = None
        best_objective = np.inf
        for i_result, result in enumerate(results):
            bit_array = self.get_bit_array(result)
            # Decode solution, i.e., evaluate nodal stress and member areas.
            member_stresses_sol = self.decode_member_stress_solution(result)
            member_areas_sol = self.decode_member_area_solution(result)
            # Compute member forces, complementary energy, and volume.
            member_forces_sol = [member_stresses_sol[i]*member_areas_sol[i] for i in range(len(member_stresses_sol))]
            complementary_energy_sol = self.complementary_energy(member_stresses_sol, member_areas_sol)
            volume_sol = self.total_volume(member_areas_sol)
            # Evaluate constraints (joint residuals and volume constraint, if any).
            joint_residuals_x_sol,  joint_residuals_y_sol, joint_residuals_squared_sum_sol = self.joint_residuals_squared_sum(member_stresses_sol, member_areas_sol)
            joint_residuals_sol = joint_residuals_x_sol + joint_residuals_y_sol
            constraints_sol = joint_residuals_sol
            volume_residual_sol= 0.0
            if hasattr(self, 'volume_constraint'):
                if self.volume_constraint['mode'] == 'direct':
                    if self.volume_constraint['type'] == 'eq':
                        volume_residual_sol = (volume_sol-self.target_volume)/self.target_volume
                    elif self.volume_constraint['type'] == 'ineq':
                         volume_residual_sol = max(0.0, (volume_sol-self.target_volume)/self.target_volume)
                elif self.volume_constraint['mode'] == 'num_add_members':
                    num_add_members = 0
                    for i_member, member in enumerate(self.members):
                        if member in self.optional_members and member_areas_sol[i_member] > 0.:
                            num_add_members += 1
                    if self.volume_constraint['type'] == 'eq':
                        res_max_members = (num_add_members-self.target_num_add_members)/self.target_num_add_members
                        volume_residual_sol = res_max_members
                    elif self.volume_constraint['type'] == 'ineq':
                        res_max_members = max(0.0, (num_add_members-self.target_num_add_members)/self.target_num_add_members)
                        volume_residual_sol = res_max_members
                constraints_sol.extend([volume_residual_sol])
            volume_residual_squared_sol = volume_residual_sol**2
            constraints_squared_sum_sol = joint_residuals_squared_sum_sol + volume_residual_squared_sol
            # Compute objective function.
            objective_sol = self.objective(complementary_energy_sol, constraints_squared_sum_sol, constraints_sol)

            if objective_sol < best_objective:
                best_solution = {
                    'bit_array': bit_array,
                    'member_forces': member_forces_sol,
                    'member_stresses': member_stresses_sol,
                    'member_areas': member_areas_sol,
                    'adaptive_vars': self.get_adaptive_vars(member_stresses_sol, member_areas_sol),
                    'complementary_energy': complementary_energy_sol,
                    'volume': volume_sol,
                    'volume_residual_squared': volume_residual_squared_sol,
                    'joint_residuals_squared_sum': joint_residuals_squared_sum_sol,
                    'constraints': constraints_sol,
                    'constraints_squared_sum': constraints_squared_sum_sol,
                    'objective': objective_sol
                }

                if hasattr(self, 'ts_ref'):
                    rel_error_forces, area_mismatch, rel_error_compliance = self.compare_with_reference_solution(best_solution)
                    best_solution['avg_rel_error_forces'] = np.nanmean(rel_error_forces)
                    best_solution['rel_error_compliance'] = rel_error_compliance
                    best_solution['areas_matching'] = not any(area_mismatch)

                best_objective = objective_sol

        output =  'Best solution (minimum objective):\n'
        output += '----------------------------------\n'
        output += f"Complementary Energy: {best_solution['complementary_energy']}\n"
        output += f"Volume: {best_solution['volume']}\n"
        output += f"Objective: {best_solution['objective']}\n"
        output += f"Constraints (squared sum): {best_solution['constraints_squared_sum']}\n"
        output += f"\tJoint Residuals (squared): {best_solution['joint_residuals_squared_sum']}\n"
        output += f"\tVolume residual (squared): {best_solution['volume_residual_squared']}\n"
        output += f"Member Forces: {best_solution['member_forces']}\n"
        output += f"Member Stresses: {best_solution['member_stresses']}\n"
        output += f"Member Areas: {best_solution['member_areas']}\n"
        if hasattr(self, 'ts_ref'):
            output += f"Compliance:\n\t Rel. error {best_solution['rel_error_compliance']}\n"
            output += f"Member Forces:\n\tAverage rel. error {best_solution['avg_rel_error_forces']}\n"
            output += f"Member Areas:\n\tMatching reference solution: {best_solution['areas_matching']}\n"
        self.print_and_log(output)


        return best_solution

    def decode_member_stress_solution(self, result):
        member_stress_sol = []
        for i_member, member_stress_poly in enumerate(self.member_stress_polys):
            if not self.members[i_member].exists:
                member_stress_sol.append(0.)
            elif type(result) is SampleView:
                member_stress_sol.append(self.decode_amplify_poly_with_bitstring(member_stress_poly,result._data))
            elif type(result) is SimpleNamespace:
                member_stress_sol.append(self.decode_amplify_poly_with_bitstring(member_stress_poly,result.values))
            else:
                member_stress_sol.append(member_stress_poly.decode(result.values))
        return member_stress_sol

    def decode_member_area_solution(self, result):
        member_area_sol = []
        for member_area_poly in self.member_area_polys:
            if isinstance(member_area_poly, Poly):
                if type(result) is SampleView:
                    member_area_sol.append(self.decode_amplify_poly_with_bitstring(member_area_poly,result._data))
                elif type(result) is SimpleNamespace:
                    member_area_sol.append(self.decode_amplify_poly_with_bitstring(member_area_poly,result.values))
                else:
                    member_area_sol.append(member_area_poly.decode(result.values))
            else:
                member_area_sol.append(member_area_poly)
        return member_area_sol

    def set_reference_solution(self, ts_ref):
        self.ts_ref = ts_ref
        # Check if reference truss structures is matching (i.e., subset of nodes and supports, same loads)
        ref_nodes_form_subset = (ts_ref.nodes.items() <= self.nodes.items())
        if not ref_nodes_form_subset:
            raise Exception('Nodes in the reference solution do not form a subset of the truss structure nodes.')
        # Compare supports
        ref_supports_form_subset = (ts_ref.supports.items() <= self.supports.items())
        if not ref_supports_form_subset:
            raise Exception('Supports in the reference solution do not form a subset of the truss structure supports.')
        # Compare loads
        loads_equal = (self.loads == ts_ref.loads)
        if not loads_equal:
            raise Exception('Loads are not matching.')

        # Check if reference solution is statically determinate
        statically_determinate_info = self.ts_ref.check_statically_determinate()
        if statically_determinate_info['condition'] != 'determinate':
            output = f"Reference solution is " \
                       f"{statically_determinate_info['condition']} " \
                       f"with degree {statically_determinate_info['degree']}.\n"
            self.print_and_log(output)

    def compare_with_reference_solution(self, solution):
        if not hasattr(self, 'ts_ref'):
            raise Exception('No reference solution set for comparison')

        # Ensure that nodes in the reference solution are matching
        for node_id, node_coords in self.ts_ref.nodes.items():
            if self.nodes[node_id] != node_coords:
                raise Exception('Nodes in the provided reference solution are not matching.')

        # Compute reference solution
        self.member_forces_ref_sol, self.compliance_ref_sol = self.ts_ref.compute_member_forces()

        # Compute relative error in compliance
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

        # Compute relative error in member forces e if all members match.
        # This only makes sense if the members in the solution and reference solution match.
        if not any(area_mismatch):
            member_forces = solution['member_forces']
            member_areas = solution['member_areas']
            active_indices = [i for i, area in enumerate(member_areas) if area > 0.0]
            filtered_member_forces = [member_forces[i] for i in active_indices]

            rel_error_forces = []
            for i_member in range(len(self.member_forces_ref_sol)):
                if self.member_forces_ref_sol[i_member] != 0:
                    rel_error_force = abs((self.member_forces_ref_sol[i_member]-filtered_member_forces[i_member])) \
                        /abs(self.member_forces_ref_sol[i_member])
                else:
                    rel_error_force = np.nan
                rel_error_forces.append(rel_error_force)
        else:

            rel_error_forces = [np.nan for _ in self.members]


        return rel_error_forces, area_mismatch, rel_error_compliance