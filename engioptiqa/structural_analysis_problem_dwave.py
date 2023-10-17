from .base_problem_dwave import BaseProblemDWave
from .structural_analysis_problem import StructuralAnalysisProblem

class StructuralAnalysisProblemDWave(BaseProblemDWave, StructuralAnalysisProblem):

    def generate_discretization(self, n_qubits_per_node, binary_representation):
        BaseProblemDWave.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation)
        StructuralAnalysisProblem.generate_cross_section_inverse_polys(self)