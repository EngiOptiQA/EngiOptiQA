import itertools
from .base_problem import BaseProblem

class StructuralAnalysisProblem(BaseProblem):
    def __init__(self, rod, g, output_path=None):
        super().__init__(rod, g, output_path)

        self.name = 'Structural Analysis Problem'

        self.print_and_log(self.name+'\n')

    def analytical_complementary_energy_and_compliance(self):
        A_combi = list(itertools.product([self.rod.A], repeat=self.rod.n_comp))
        super().analytical_complementary_energy_and_compliance(A_combi)

    def compute_analytical_solution(self):
        self.analytical_complementary_energy_and_compliance()
        self.A_analytic = self.rod.cross_sections
        self.PI_analytic = self.PI_combi[0]
        self.C_analytic = self.C_combi[0]
        self.stress_analytic = self.compute_stress_function(self.rod)
        self.force_analytic = self.compute_force_function(self.stress_analytic, self.rod)
        self.displacement_analytic = self.compute_displacement_function(self.stress_analytic, self.rod)

        output = f'Analytic Force: {self.force_analytic}\n'
        self.print_and_log(output)

    def generate_discretization(self, n_qubits_per_node, binary_representation, lower_lim=None, upper_lim=None):
        BaseProblem.initialize_discretization(self)
        BaseProblem.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation,  lower_lim, upper_lim)
        self.generate_cross_section_inverse_polys()

    def generate_cross_section_inverse_polys(self):
        cs_inv_polys = []
        for _ in range(self.rod.n_comp):
            cs_inv_polys.append(1./self.rod.A)
        self.cs_inv_polys = cs_inv_polys

