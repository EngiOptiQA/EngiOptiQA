from openjij import SASampler

from .annealing_solver import AnnealingSolver

class AnnealingSolverOpenJij(AnnealingSolver):
    def __init__(self):
        self.client_type = 'openjij'
        super().__init__()

    def setup_client(self):
        self.client = None

    def setup_solver(self):
        self.solver = SASampler()

    def solve_problem(self, problem, **kwargs):
        problem.results = self.solver.sample_hubo(problem.binary_model.objective.as_dict(), "BINARY", **kwargs)
