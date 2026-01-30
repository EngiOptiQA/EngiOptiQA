from amplify import (
    Model
)

from .truss_structure import TrussStructure

class TrussStructureOptimization(TrussStructure):
    def generate_member_area_inv_polys(self):
        assert(self.variable_generator is not None)
        print("TrussStructureOptimization: Generating member area inverse polynomials.")
        A_0 = 0.
        A_1 = 0.5
        member_area_polys = []
        member_area_inv_polys = []
        for _ in range(self.n_members):
            q = self.variable_generator.array("Binary", 1)
            member_area_polys.append(A_0 + (A_1-A_0)*q[0])
            #member_area_inv_polys.append(1./A_0 + (1./A_1-1./A_0)*q[0])
            member_area_inv_polys.append((1./A_1)*q[0])
        self.member_area_polys = member_area_polys
        self.member_area_inv_polys = member_area_inv_polys

    def update_member_area_inv_polys(self):
        # self.initialize_discretization()
        A_0 = 0.
        A_1 = 0.5
        member_area_polys = []
        member_area_inv_polys = []
        for _ in range(self.n_members):
            q = self.variable_generator.array("Binary", 1)
            member_area_polys.append(A_0 + (A_1-A_0)*q[0])
            #member_area_inv_polys.append(1./A_0 + (1./A_1-1./A_0)*q[0])
            member_area_inv_polys.append((1./A_1)*q[0])
        self.member_area_polys = member_area_polys
        self.member_area_inv_polys = member_area_inv_polys

    def generate_objective(self, penalty_weight_joints, penalty_weight_volume, target_volume):
        super().generate_complementary_energy_poly()
        super().generate_joint_residuals_poly()
        super().generate_volume_constraint_poly(target_volume)
        self.penalty_weight_joints = penalty_weight_joints
        self.penalty_weight_volume = penalty_weight_volume
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_joints * self.joint_residuals_poly + \
            self.penalty_weight_volume * self.volume_poly

        self.binary_quadratic_model = Model(self.poly)

    def update_penalty_weight_in_objective(self, penalty_weight_joints, penalty_weight_volume, target_volume):
        self.penalty_weight_joints = penalty_weight_joints
        self.penalty_weight_volume = penalty_weight_volume
        self.poly = self.complementary_energy_poly + \
            self.penalty_weight_joints * self.joint_residuals_poly + \
            self.penalty_weight_volume * self.volume_poly

        self.binary_quadratic_model = Model(self.poly)
