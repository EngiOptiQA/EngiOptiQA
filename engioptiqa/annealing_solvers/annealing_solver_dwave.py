from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler

from .annealing_solver import AnnealingSolver

class AnnealingSolverDWave(AnnealingSolver):

    def __init__(self, token_file=None, proxy=None):
        self.client_type = 'dwave'
        super().__init__(token_file, proxy)

    def setup_client(self):

        self.client = Client(token=self.token, proxy=self.proxy)
        print("Available QPU solvers:")
        for solver in self.client.get_solvers(qpu=True, online=True):
            print("\t", solver)
        print("Available hybrid solvers:")
        for solver in self.client.get_solvers(name__regex='hybrid_binary.*', online=True):
            print("\t", solver)

    def setup_solver(self, solver_type=None, solver_name=None):
        
        if solver_name:
            self.solver_name = solver_name
            solver = self.client.get_solver(name=self.solver_name)
            if solver.qpu:
                self.solver_type = 'qpu'
                self.solver = EmbeddingComposite(
                    DWaveSampler(
                        token=self.token, 
                        proxy=self.proxy, 
                        solver=dict(name=self.solver_name)
                    )
                )
            else:
                self.solver_type = 'hybrid'
                self.solver = LeapHybridSampler(token=self.token, proxy=self.proxy)

        elif solver_type:
            self.solver_type = solver_type
            if self.solver_type == 'qpu':
                self.solver = EmbeddingComposite(
                    DWaveSampler(token=self.token, proxy=self.proxy)
                )
                self.solver_name = self.solver.child.properties["chip_id"]
             
            elif self.solver_type == 'hybrid':
                self.solver = LeapHybridSampler(token=self.token, proxy=self.proxy)
                self.solver_name =  self.solver.properties["category"] + ' ' + self.solver.properties["version"]

            elif self.solver_type == 'simulated_annealing':
                self.solver = SimulatedAnnealingSampler() 
                self.solver_name = 'simulated annealing'
            else:
                raise Exception('Unknown solver type', self.solver_type)
        else:
            raise Exception('Either a solver type or a specific solver name must be specified.')
        
        print("Use", self.solver_type, "solver:", self.solver_name)

    def solve_qubo_problem(self, problem, **kwargs):

        problem.results_indices = self.solver.sample(
            problem.binary_quadratic_model_indices, 
            **kwargs
        )
        print('Number of solutions:', len(problem.results_indices))
        if hasattr(problem, 'mapping_i_to_q'):
            problem.results =  problem.results_indices.relabel_variables(
                problem.mapping_i_to_q,
                inplace=False)
        else:
            problem.results = problem.results_indices 

    def perform_local_search(self, problem):

        if hasattr(problem, 'results_indices'):
            solver_greedy = SteepestDescentSolver()
            sampleset_pp = solver_greedy.sample(
                problem.binary_quadratic_model_indices,
                initial_states=problem.results_indices
                ) 
            if hasattr(problem, 'mapping_i_to_q'):
                problem.results_pp = sampleset_pp.relabel_variables(
                    problem.mapping_i_to_q,
                    inplace=False)
            else:
                problem.results_pp = sampleset_pp
            
        else:
            raise Exception('Trying to perform local search altough no results exist yet.')
