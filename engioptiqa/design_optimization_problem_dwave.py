from pyqubo import Array

from .base_problem_dwave import BaseProblemDWave
from .design_optimization_problem import DesignOptimizationProblem

class DesignOptimizationProblemDWave(BaseProblemDWave, DesignOptimizationProblem):

    def __init__(self, rod, g, A_choice, output_path=None):
        raise Exception(
            "Using design optimization problems created"
            "with DWave SDK not yet supported."
            )  
          
    def generate_discretization(self, n_qubits_per_node, binary_representation):
        BaseProblemDWave.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation)
        self.generate_cross_section_polys()

    def generate_cross_section_polys(self):
        cs_inv_polys = []
        q = Array.create('q_A', shape=(self.rod.n_comp), vartype='BINARY')
        for i_comp in range(self.rod.n_comp):
            cs_inv_polys.append(1./self.A_choice[0] + (1./self.A_choice[1]-1./self.A_choice[0])*q[i_comp])
        self.cs_inv_polys = cs_inv_polys