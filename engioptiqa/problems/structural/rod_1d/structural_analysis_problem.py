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

    def get_initial_number_of_problem_variables(self):
        return self.rod.n_comp * self.n_qubits_per_var

    def generate_cross_section_inverse_poly(self, i_comp):
        return 1./self.rod.A

