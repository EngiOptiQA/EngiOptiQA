import itertools
import numpy as np
from .base_problem import BaseProblemRod1D

class StructuralAnalysisProblemRod1D(BaseProblemRod1D):
    def __init__(self, rod, g, output_path=None):
        super().__init__(rod, g, output_path)

        self.name = 'Structural Analysis Problem'

        self.print_and_log(self.name+'\n')

    def get_analytical_cross_sections(self):
        return np.ones(self.rod.n_comp)*self.rod.A

    def generate_discretization(self, n_qubits_per_var, binary_representation, lower_lim=None, upper_lim=None):
        BaseProblemRod1D.initialize_discretization(self)
        BaseProblemRod1D.generate_nodal_force_polys(self, n_qubits_per_var, binary_representation, lower_lim, upper_lim)
        self.generate_cross_section_inverse_polys()

    def generate_cross_section_inverse_polys(self):
        cs_inv_polys = []
        for _ in range(self.rod.n_comp):
            cs_inv_polys.append(1./self.rod.A)
        self.cs_inv_polys = cs_inv_polys

