from amplify import Model
import numpy as np
from engioptiqa.variables.real_number import RealNumber
from .truss_structure import TrussStructure

class TrussStructureOptimization(TrussStructure):
    def __init__(self, target_volume=None, volume_constraint='equality', output_path=None):
        super().__init__(output_path)
        self.target_volume = target_volume
        self.volume_constraint = volume_constraint

    def generate_member_area_polys(self):
        assert(self.variable_generator is not None)
        member_area_polys = []
        member_areas = self.get_member_areas()

        q = self.variable_generator.array("Binary", self.n_optional_members , name=f"q_A")
        i_optional_member = 0
        for i_member in range(self.n_members):
            A = member_areas[i_member]
            if self.members[i_member] in self.optional_members:
                member_area_polys.append(A*q[i_optional_member])
                i_optional_member += 1
            else:
                member_area_polys.append(A)

        self.member_area_polys = member_area_polys

    def set_target_volume(self, target_volume):
        self.target_volume = target_volume

    def generate_slack_variable(self, n_qubits_slack=3):
        assert(self.variable_generator is not None)
        q = self.variable_generator.array("Binary", n_qubits_slack , name=f"s_vol")
        s = RealNumber(n_qubits_slack, 'normalized')
        self.slack_variable = s.evaluate(q)

    def generate_volume_constraint_poly(self):

        member_areas = self.member_area_polys
        if self.target_volume is not None and self.target_volume > 0.:
            con_volume = (self.total_volume(member_areas)-self.target_volume)/self.target_volume
            if self.volume_constraint == 'equality':
                self.volume_con_poly = [con_volume]
                self.volume_con_squared_poly = con_volume**2
            elif self.volume_constraint == 'inequality':
                s = self.slack_variable
                volume_ratio = self.total_volume(member_areas)/self.target_volume
                self.volume_con_poly = [volume_ratio+s-1.]
                self.volume_con_squared_poly = (volume_ratio+s-1.)**2
        else:
            raise Exception("Target volume must be set and be greater than zero.")


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
    def __init__(self, target_volume=None, volume_constraint='equality', output_path=None):
        super().__init__(target_volume=target_volume, volume_constraint=volume_constraint, output_path=output_path)

    def generate_discretization(self,
                                n_qubits_per_var, binary_representation_stress,
                                n_qubits_per_area, binary_representation_area,
                                lower_lim_stress=None, upper_lim_stress=None,
                                lower_lim_area=None, upper_lim_area=None):
        self.initialize_discretization()
        self.generate_member_stress_polys(n_qubits_per_var, binary_representation_stress, lower_lim_stress, upper_lim_stress)
        self.generate_member_area_polys(n_qubits_per_area, binary_representation_area, lower_lim_area, upper_lim_area)
        if self.volume_constraint == 'inequality':
            self.generate_slack_variable()

    def has_adaptive_variables(self):
        return (self.binary_representation == 'adaptive_range') or (self.binary_representation_area == 'adaptive_range')

    def get_number_of_adaptive_vars(self):
        if self.binary_representation == 'adaptive_range':
            if self.binary_representation_area == 'adaptive_range':
                return np.array([self.n_members, self.n_optional_members])
            else:
                return np.array([self.n_members])
        else:
            return np.array([0])

    def get_adaptive_vars(self, member_stresses_sol, member_areas_sol):
        if self.binary_representation == 'adaptive_range':
            if self.binary_representation_area == 'adaptive_range':
                return [member_stresses_sol, member_areas_sol]
            else:
                return [member_stresses_sol]
        else:
            return None

    def get_position_in_bit_array(self, i_group, i_var):

        n_existent_members = self.get_number_of_existent_members()
        if i_group == 0:
            start = i_var * self.n_qubits_per_var
            end = (i_var + 1) * self.n_qubits_per_var
        elif i_group == 1:
            start = n_existent_members * self.n_qubits_per_var + i_var * self.n_qubits_per_area
            end = n_existent_members * self.n_qubits_per_var + (i_var + 1) * self.n_qubits_per_area
        else:
            raise Exception(f"Invalid group index: {i_group}.")
        return start, end

    def get_real_number_object(self, i_group):
        if i_group == 0:
            return self.real_number
        elif i_group == 1:
            return self.real_number_areas
        else:
            raise Exception(f"Invalid group index: {i_group}.")

    def get_range_limits(self, i_group):
        assert(i_group < 2)
        if i_group == 0:
            return self.a_min, self.a_max
        elif i_group == 1:
            return self.A_min, self.A_max
        else:
            return None, None

    def update_formulation(self, best_solution):
        self.update_member_stress_polys()
        self.update_member_area_polys(best_solution)
        if self.volume_constraint == 'inequality':
            self.generate_slack_variable()

    def generate_member_area_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        assert(self.variable_generator is not None)
        if binary_representation in ['range', 'adaptive_range']:
            assert(lower_lim is not None and upper_lim is not None), \
                "Lower and upper limits must be provided for range representation."
            self.A_min = [lower_lim] * self.n_members
            self.A_max = [upper_lim] * self.n_members
        self.n_qubits_per_area = n_qubits_per_var
        self.binary_representation_area = binary_representation
        self.real_number_areas = RealNumber(self.n_qubits_per_area, self.binary_representation_area, lower_lim, upper_lim, a_min_lim=0.0)

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

    def update_member_area_polys(self, best_solution):
        member_area_polys = []
        member_areas = self.get_member_areas()
        i_optional_member = 0
        i_active_member = 0
        for i_member, _ in enumerate(self.members):
            if self.members[i_member] in self.optional_members:
                if not self.members[i_member].exists:
                    member_area_polys.append(0.)
                else:
                    A = best_solution["member_areas"][i_member]
                    if A < 1e-2*self.members[i_member].A_initial:
                        member_area_polys.append(0.)
                        self.members[i_member].exists = False
                    else:
                        q = self.variable_generator.array("Binary", self.n_qubits_per_area, name=f"q_A_{i_optional_member}")
                        if self.binary_representation_area == 'adaptive_range':
                            self.real_number_areas.set_range(self.A_min[i_member], self.A_max[i_member])
                        member_area_polys.append(self.real_number_areas.evaluate(q))
                    i_active_member += 1
                i_optional_member += 1
            else:
                A = member_areas[i_member]
                member_area_polys.append(A)
        self.member_area_polys = member_area_polys
