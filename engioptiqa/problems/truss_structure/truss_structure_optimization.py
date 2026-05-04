from amplify import (Model)
import numpy as np
from engioptiqa.variables.real_number import RealNumber
from .truss_structure import TrussStructure

class TrussStructureOptimization(TrussStructure):
    def __init__(self, target_volume=None, output_path=None):
        super().__init__(output_path)
        self.target_volume = target_volume

    # def generate_discretization(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None, lower_lim_2=None, upper_lim_2=None):
    #     self.initialize_discretization()
    #     self.generate_member_stress_polys(n_qubits_per_var, binary_representation, lower_lim, upper_lim)
    #     self.generate_member_area_polys(n_qubits_per_var, binary_representation, lower_lim_2, upper_lim_2)

    # def update_formulation(self):
    #     self.update_member_stress_polys()
    #     self.update_member_area_polys()

    # def generate_member_area_polys(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
    def generate_member_area_polys(self):
        assert(self.variable_generator is not None)
        member_area_polys = []
        member_areas = self.get_member_areas()

        # if binary_representation == 'range':
        #     assert(lower_lim is not None and upper_lim is not None), \
        #         "Lower and upper limits must be provided for range representation."
        #     self.b_min = np.ones(len(self.members))*lower_lim
        #     self.b_max = np.ones(len(self.members))*upper_lim
        # self.n_qubits_per_var = n_qubits_per_var
        # self.binary_representation = binary_representation
        # self.real_number_2 = RealNumber(self.n_qubits_per_var, self.binary_representation, lower_lim, upper_lim)


        for i_member in range(self.n_members):
            q = self.variable_generator.array("Binary", 1)
            A = member_areas[i_member]
            member_area_polys.append(A*q[0])

            # q = self.variable_generator.array("Binary", self.n_qubits_per_var)
            # if self.binary_representation == 'range':
            #     self.real_number_2.set_range(self.b_min[i_member], self.b_max[i_member])
            # member_area_polys.append(self.real_number_2.evaluate(q))

        self.member_area_polys = member_area_polys

    # def update_member_area_polys(self):
    #     member_area_polys = []
    #     for i_member, _ in enumerate(self.members):
    #         q = self.variable_generator.array("Binary", self.n_qubits_per_var)
    #         if self.binary_representation == 'range':
    #             self.real_number_2.set_range(self.b_min[i_member], self.b_max[i_member])
    #         member_area_polys.append(self.real_number_2.evaluate(q))
    #     self.member_area_polys = member_area_polys

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
                for poly in self.member_area_polys
            ]
            overall_mean_of_means = float(np.mean(means)) if means else np.nan
            con_volume = self.total_volume(member_areas)/overall_mean_of_means
        self.volume_poly = con_volume**2

    def generate_problem_formulation(self, penalty_weight):#, penalty_weight_volume, target_volume):
        super().generate_complementary_energy_poly()
        super().generate_joint_residuals_poly()
        self.generate_volume_constraint_poly()
        self.penalty_weight = penalty_weight
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight * ( self.joint_residuals_poly + self.volume_poly)

        self.binary_model = Model(self.poly)


