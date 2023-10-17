from amplify import QuadratizationMethod
from .base_problem_amplify import BaseProblemAmplify
from .design_optimization_problem import DesignOptimizationProblem

class DesignOptimizationProblemAmplify(BaseProblemAmplify, DesignOptimizationProblem):
    
    def generate_discretization(self, n_qubits_per_node, binary_representation):
        BaseProblemAmplify.initialize_discretization(self)
        BaseProblemAmplify.generate_nodal_force_polys(self, n_qubits_per_node, binary_representation)
        self.generate_cross_section_polys()

    def generate_cross_section_polys(self):
        assert(self.symbol_generator is not None)
        cs_inv_polys = []

        for _ in range(self.rod.n_comp):
            q = self.symbol_generator.array(1)
            cs_inv_polys.append(1./self.A_choice[0] + (1./self.A_choice[1]-1./self.A_choice[0])*q[0])
        self.cs_inv_polys = cs_inv_polys

    def set_quad_method(self,quad_method_name):
        self.quad_method_name = quad_method_name
        match quad_method_name:
            case 'ISHIKAWA':
                self.quad_method = QuadratizationMethod.ISHIKAWA
            case 'ISHIKAWA_KZFD':
                self.quad_method = QuadratizationMethod.ISHIKAWA_KZFD
            case 'SUBSTITUTION':
                self.quad_method = QuadratizationMethod.SUBSTITUTION
            case 'SUBSTITUTION_KZFD':
                self.quad_method = QuadratizationMethod.SUBSTITUTION_KZFD
            case _ :
                raise Exception('Unknown quad_method', quad_method_name)