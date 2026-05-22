from amplify import Model
import numpy as np
from engioptiqa.variables.real_number import RealNumber
from .truss_structure import TrussStructure

class TrussStructureOptimization(TrussStructure):
    def __init__(self, target_volume=None, output_path=None):
        super().__init__(output_path)
        self.target_volume = target_volume

    def generate_member_area_polys(self):
        assert(self.variable_generator is not None)
        member_area_polys = []
        member_areas = self.get_member_areas()

        q = self.variable_generator.array("Binary", self.n_optional_members , name=f"q_A")
        i_optional_member = 0
        for i_member in range(self.n_members):
            if self.members[i_member] in self.optional_members:
                A = member_areas[i_member]
                member_area_polys.append(A*q[i_optional_member])
                i_optional_member += 1
            else:
                A = member_areas[i_member]
                member_area_polys.append(A)

        self.member_area_polys = member_area_polys

    def set_target_volume(self, target_volume):
        self.target_volume = target_volume

    def generate_volume_constraint_poly(self):
        member_areas = self.member_area_polys
        if self.target_volume > 0.:
            con_volume = (self.total_volume(member_areas)-self.target_volume)/self.target_volume
        else:
            # Compute overall mean coefficient for scaling
            means = [
                np.fromiter(poly.as_dict().values(), dtype=float).mean()
                if hasattr(poly, "as_dict") and callable(poly.as_dict) else np.nan
                for poly in self.member_area_polys
                ]

            overall_mean_of_means = float(np.mean(means)) if means else np.nan
            con_volume = self.total_volume(member_areas)/overall_mean_of_means
        self.volume_con_poly = [con_volume]
        self.volume_con_squared_poly = con_volume**2

    def generate_constraint_polys(self):
        self.generate_joint_residuals_poly()
        self.generate_volume_constraint_poly()
        self.constraint_polys = self.joint_residual_polys + self.volume_con_poly
        self.constraints_sum_squared_poly = self.joint_residuals_squared_sum_poly + self.volume_con_squared_poly

    def get_n_constraints(self):
        return super().get_n_constraints() + 1

    def generate_problem_formulation(self, penalty_weight=1.0, lagrange_multipliers=[], mode='penalty'):
        super().generate_complementary_energy_poly()
        self.generate_constraint_polys()
        super().generate_objective_poly(penalty_weight=penalty_weight, lagrange_multipliers=lagrange_multipliers, mode=mode)

        self.binary_model = Model(self.poly)

class TrussStructureOptimizationContinuous(TrussStructureOptimization):
    def __init__(self, target_volume=None, output_path=None):
        super().__init__(target_volume=target_volume, output_path=output_path)

    def generate_discretization(self,
                                n_qubits_per_var, binary_representation_stress,
                                n_qubits_per_area, binary_representation_area,
                                lower_lim_stress=None, upper_lim_stress=None,
                                lower_lim_area=None, upper_lim_area=None):
        self.initialize_discretization()
        self.generate_member_stress_polys(n_qubits_per_var, binary_representation_stress, lower_lim_stress, upper_lim_stress)
        self.generate_member_area_polys(n_qubits_per_area, binary_representation_area, lower_lim_area, upper_lim_area)


    def get_number_of_adaptive_vars(self):
        if self.binary_representation == 'adaptive_range':
            if self.binary_representation_area == 'adaptive_range':
                return self.n_members + self.n_optional_members
            else:
                return self.n_members
        else:
            return 0

    def get_adaptive_vars(self, member_stresses_sol, member_areas_sol):
        return member_stresses_sol + member_areas_sol

    def get_position_in_bit_array(self, i_var):

        if i_var < self.n_members:
            start = i_var * self.n_qubits_per_var
            end = (i_var + 1) * self.n_qubits_per_var
        elif i_var >= self.n_members and i_var < 2*self.n_members:
            i_area_var = i_var - self.n_members
            start = self.n_members * self.n_qubits_per_var + i_area_var * self.n_qubits_per_area
            end = self.n_members * self.n_qubits_per_var + (i_area_var + 1) * self.n_qubits_per_area
        else:
            raise Exception(f"Invalid variable index: {i_var}. Must be between 0 and {2*self.n_members-1}.")
        return start, end

    def update_formulation(self):
        self.update_member_stress_polys()
        self.update_member_area_polys()

    def generate_member_area_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation in ['range', 'adaptive_range']:
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.a_min = np.concatenate([self.a_min] + [[lower_lim] * self.n_members])
            self.a_max = np.concatenate([self.a_max] + [[upper_lim] * self.n_members])
        self.n_qubits_per_area = n_qubits_per_var
        self.binary_representation_area = binary_representation
        self.real_number_areas = RealNumber(self.n_qubits_per_area, self.binary_representation_area, lower_lim, upper_lim)

        member_area_polys = []
        member_areas = self.get_member_areas()
        for i_member, _ in enumerate(self.members):
            i_optional_member = 0
            if self.members[i_member] in self.optional_members:
                q = self.variable_generator.array("Binary", self.n_qubits_per_area, name=f"q_A_{i_optional_member}")
                member_area_polys.append(self.real_number_areas.evaluate(q))
                i_optional_member += 1
            else:
                A = member_areas[i_member]
                member_area_polys.append(A)
        self.member_area_polys = member_area_polys

    def update_member_area_polys(self):
        member_area_polys = []
        member_areas = self.get_member_areas()
        for i_member, _ in enumerate(self.members):
            i_optional_member = 0
            if self.members[i_member] in self.optional_members:
                q = self.variable_generator.array("Binary", self.n_qubits_per_area, name=f"q_A_{i_optional_member}")
                if self.binary_representation_area == 'adaptive_range':
                    self.real_number_areas.set_range(self.a_min[self.n_members + i_member], self.a_max[self.n_members + i_member])
                member_area_polys.append(self.real_number_areas.evaluate(q))
                i_optional_member += 1
            else:
                A = member_areas[i_member]
                member_area_polys.append(A)
        self.member_area_polys = member_area_polys
