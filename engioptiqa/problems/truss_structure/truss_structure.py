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
        super().__init__(output_path)
        self.nodes = {}  # Dictionary to store nodes: {node_id: (x, y)}
        self.n_nodes = 0
        self.members = []  # List to store truss members
        self.n_members = 0
        self.loads = {}  # Dictionary to store external forces: {node_id: (Fx, Fy)}
        self.supports = {}  # Dictionary to store support conditions: {node_id: (x_fixed, y_fixed)}

        self.penalty_weight_joints = 0.0
        self.penalty_weight_volume = 0.0

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
        :param truss: Instance of TrussStructure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot nodes
        for node_id, (x, y) in self.nodes.items():
            ax.plot(x, y, 'o', color='green')  # Nodes as black circles
            ax.text(x - 0.075, y - 0.075, f"{node_id}", fontsize=12)

        # Plot members
        A_max = max([member.A for member in self.members if member.A is not None]) if any(member.A is not None for member in self.members) else 1.0
        # print(f'Max Area: {A_max}')

        for member in self.members:
            x0, y0 = member.get_coords(local_node_id = 0)
            x1, y1 = member.get_coords(local_node_id = 1)
            lw = member.A / A_max * 5 if member.A is not None else 1.0
            ax.plot([x0, x1], [y0, y1], 'k-',lw=lw, label="Member")  # Members as black lines

        # Plot loads
        for node_id, (Fx, Fy) in self.loads.items():
            x, y = self.nodes[node_id]
            F_norm = (Fx**2 + Fy**2)**0.5
            ax.arrow(x, y, Fx / (2*F_norm), Fy / (2*F_norm), color='red', head_width=0.025, length_includes_head=True)  # Loads as red arrows

        # Plot supports
        for node_id, (x_fixed, y_fixed) in self.supports.items():
            x, y = self.nodes[node_id]
            if x_fixed and y_fixed:
                ax.plot(x, y, 's', color='blue')  # Fixed supports as blue squares
            elif x_fixed:
                ax.plot(x, y, '>', color='blue')  # Pinned supports (x) as blue triangles
            elif y_fixed:
                ax.plot(x, y, '^', color='blue')  # Pinned supports (y) as blue triangles

        # Custom legend entries
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Members'),
            Line2D([0], [0], marker='o', color='green', markersize=8, label='Nodes', linestyle='None'),
            Line2D([0], [0], marker='s', color='blue', markersize=8, label='Supports', linestyle='None'),
            Line2D([0], [0], color='red', lw=2, label='Loads (Scaled)'),
        ]

        # Add legend
        ax.legend(handles=legend_elements, loc="upper right")

        # Set plot properties
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Truss Structure: " + subtitle)
        plt.show()

    def compute_member_forces(self):
        """
        Compute axial forces in the truss members using the method of joints.
        :return: Dictionary of member forces: {(node_1_id, node_2_id): force}.
        """
        # Number of nodes and members
        num_nodes = len(self.nodes)
        num_members = len(self.members)

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
        # print("Global Force Vector:", global_force)
        # print("Nodal Displacements:", displacements)
        compliance = np.inner(global_force, displacements)
        print("Compliance:", compliance)

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
        #self.generate_member_force_polys(n_qubits_per_var, binary_representation, lower_lim, upper_lim)
        self.generate_member_stress_polys(n_qubits_per_var, binary_representation, lower_lim, upper_lim)
        self.generate_member_area_inv_polys()

    def get_number_of_continuous_vars(self):
        return self.n_members

    def update_formulation(self):
        self.update_member_stress_polys()
        self.generate_member_area_inv_polys()

    def generate_member_force_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation == 'range':
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.ones(len(self.members))*lower_lim
            self.a_max = np.ones(len(self.members))*upper_lim
        self.n_qubits_per_var = n_qubits_per_var
        self.binary_representation = binary_representation
        self.real_number = RealNumber(self.n_qubits_per_var, self.binary_representation, lower_lim, upper_lim)

        member_force_polys = []
        for i_member, member in enumerate(self.members):
            # print(f'member {i_member} ({member.node_id_0}, {member.node_id_1})')
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            member_force_polys.append(self.real_number.evaluate(q))
        self.member_force_polys = member_force_polys

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
        for i_member, member in enumerate(self.members):
            # print(f'member {i_member} ({member.node_id_0}, {member.node_id_1})')
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            member_stress_polys.append(self.real_number.evaluate(q))
        self.member_stress_polys = member_stress_polys

    def update_member_stress_polys(self):
        self.initialize_discretization()
        member_stress_polys = []
        for i_member, member in enumerate(self.members):
            q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            if self.binary_representation == 'range':
                self.real_number.set_range(self.a_min[i_member], self.a_max[i_member])
            member_stress_polys.append(self.real_number.evaluate(q))
        self.member_stress_polys = member_stress_polys

    def generate_member_area_inv_polys(self):
        member_area_polys = []
        member_area_inv_polys = []
        for member in self.members:
            member_area_polys.append(member.A)
            member_area_inv_polys.append(1./member.A)
        self.member_area_polys = member_area_polys
        self.member_area_inv_polys = member_area_inv_polys

    def complementary_energy(self, member_forces, member_stresses, member_areas, member_areas_inv):
        U = []
        for i_member, member in enumerate(self.members):
            #F = member_forces[i_member]
            A = member_areas[i_member]
            A_inv = member_areas_inv[i_member]
            L = member.length
            E = member.E
            stress = member_stresses[i_member]
            #U.append((F**2*L)/(2*E)*A_inv)
            U.append(stress**2* A * L / (2*E))
        V = [0 for _ in range(len(self.members))]
        # Total Complementary Energy.
        PI = sum(U + V)
        return PI

    def generate_complementary_energy_poly(self):
        #member_forces = self.member_force_polys
        member_forces = None
        member_stresses = self.member_stress_polys
        member_areas = self.member_area_polys
        member_areas_inv = self.member_area_inv_polys

        PI_poly = self.complementary_energy(member_forces, member_stresses, member_areas, member_areas_inv)

        self.complementary_energy_poly = PI_poly


    def total_volume(self, member_areas):
        total_volume = 0.0
        for i_member, member in enumerate(self.members):
            L = member.length
            A = member_areas[i_member]
            total_volume += A * L
        return total_volume

    def generate_volume_constraint_poly(self, target_volume):

        member_areas = self.member_area_polys
        # self.volume_poly = (self.total_volume(member_areas)-target_volume)**2
        self.volume_poly = (self.total_volume(member_areas))**2

    def joint_residuals_squared(self, member_forces, member_stresses, member_areas):

        # Equilibrium

        # Traction boundary conditions.
        joint_forces_x = [0. for _ in range(len(self.nodes))]
        joint_forces_y = [0. for _ in range(len(self.nodes))]
        # print('=== LOADS ===')
        for node_id, (Fx, Fy) in self.loads.items():
            joint_forces_x[node_id] += Fx
            joint_forces_y[node_id] += Fy

        # for i_node in range(self.n_nodes):
            # print(f'Node {i_node}')
            # print(f'\tx: {joint_forces_x[i_node]}')
            # print(f'\ty: {joint_forces_y[i_node]}')

        # print('=== MEMBERS ===')
        for i_member, member in enumerate(self.members):
            # print(f'Member {i_member}: {member.node_id_0} -- {member.node_id_1}')
            # print('----------------')

            node_id_0 = member.node_id_0
            node_id_1 = member.node_id_1
            #F = member_forces[i_member]
            F = member_stresses[i_member] * member_areas[i_member]
            # Get direction cosines
            l, m = member.direction_cosines_0
            joint_forces_x[node_id_0] += F * l
            joint_forces_y[node_id_0] += F * m

            l, m = member.direction_cosines_1
            joint_forces_x[node_id_1] += F * l
            joint_forces_y[node_id_1] += F * m

            # for i_node in range(self.n_nodes):
            #     print(f'Node {i_node}')
            #     print(f'\tx: {joint_forces_x[i_node]}')
            #     print(f'\ty: {joint_forces_y[i_node]}')

        cons_bc = 0.0
        for i_node in range(self.n_nodes):
            # print(f'Node {i_node}')
            x_fixed = y_fixed = False
            if i_node in self.supports.keys():
                x_fixed, y_fixed = self.supports[i_node]
            if not x_fixed:
                # print(f'\tResidual force in x-direction: {joint_forces_x[i_node]}')
                cons_bc += joint_forces_x[i_node]**2
            if not y_fixed:
                # print(f'\tResidual force in y-direction:  {joint_forces_y[i_node]}')
                cons_bc += joint_forces_y[i_node]**2


        return cons_bc

    def generate_joint_residuals_poly(self):

        #member_forces = self.member_force_polys
        member_forces = None
        member_stresses = self.member_stress_polys
        member_areas = self.member_area_polys
        self.joint_residuals_poly = self.joint_residuals_squared(member_forces, member_stresses, member_areas)

    def generate_problem_formulation(self, penalty_weight_joints):
        self.generate_complementary_energy_poly()
        self.generate_joint_residuals_poly()
        self.penalty_weight_joints = penalty_weight_joints
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_joints * self.joint_residuals_poly

        self.binary_model = Model(self.poly)

    def update_penalty_weight_in_objective(self, penalty_weight_joints):
        self.penalty_weight_joints = penalty_weight_joints
        print(f"Penalty weight (joints): {self.penalty_weight_joints}\n")
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_joints * self.joint_residuals_poly

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
            #member_forces_sol = self.decode_member_force_solution(result)
            member_forces_sol = None
            member_stresses_sol = self.decode_member_stress_solution(result)
            member_areas_sol, member_areas_inv_sol = self.decode_member_area_solution(result)
            member_forces_sol = [member_stresses_sol[i]*member_areas_sol[i] for i in range(len(member_stresses_sol))]
            complementary_energy_sol = self.complementary_energy(member_forces_sol, member_stresses_sol, member_areas_sol, member_areas_inv_sol)
            volume_sol = self.total_volume(member_areas_sol)
            joint_residuals_squared_sol = self.joint_residuals_squared(member_forces_sol, member_stresses_sol, member_areas_sol)
            objectives_sol = complementary_energy_sol + self.penalty_weight_joints * joint_residuals_squared_sol + self.penalty_weight_volume * volume_sol
            if best_solution is None or objectives_sol < best_solution['objective']:
                best_solution = solutions[i_result]

            solutions[i_result]['bit_array'] = bit_array
            solutions[i_result]['member_forces'] = member_forces_sol
            solutions[i_result]['continuous_vars'] = member_stresses_sol
            solutions[i_result]['member_areas'] = member_areas_sol
            solutions[i_result]['complementary_energy'] = complementary_energy_sol
            solutions[i_result]['volume'] = volume_sol
            solutions[i_result]['joint_residuals_squared'] = joint_residuals_squared_sol
            solutions[i_result]['objective'] = objectives_sol


        print('Best solution (minimum objective):')
        print(f"Objective: {best_solution['objective']}")
        print(f"Complementary Energy: {best_solution['complementary_energy']}")
        print(f"Volume: {best_solution['volume']}")
        print(f"Joint Residuals (squared): {best_solution['joint_residuals_squared']}")
        print(f"Member Forces: {best_solution['member_forces']}")
        print(f"Member Stresses: {best_solution['continuous_vars']}")
        print(f"Member Areas: {best_solution['member_areas']}")

        return solutions

    def decode_member_force_solution(self, result):
        member_force_sol = []
        for member_force_poly in self.member_force_polys:
            member_force_sol.append(member_force_poly.decode(result.values))
        return member_force_sol

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
        member_area_inv_sol = []
        for member_area_inv_poly in self.member_area_inv_polys:
            if isinstance(member_area_inv_poly, Poly):
                if type(result) is SampleView:
                    member_area_inv_sol.append(self.decode_amplify_poly_with_bitstring(member_area_inv_poly,result._data))
                else:
                    member_area_inv_sol.append(member_area_inv_poly.decode(result.values))
            else:
                member_area_inv_sol.append(member_area_inv_poly)
        return member_area_sol, member_area_inv_sol

