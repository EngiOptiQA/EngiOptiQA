from .base_problem_amplify import BaseProblemAmplify
from .structural_analysis_problem import StructuralAnalysisProblem

class StructuralAnalysisProblemAmplify(BaseProblemAmplify, StructuralAnalysisProblem):

    def generate_discretization(self, n_qubits_per_node, binary_representation):
        BaseProblemAmplify.initialize_discretization(self)
        BaseProblemAmplify.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation)
        StructuralAnalysisProblem.generate_cross_section_inverse_polys(self)