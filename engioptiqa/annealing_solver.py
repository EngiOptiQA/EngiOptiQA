from abc import ABC, abstractmethod

from amplify import Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient, LeapHybridSamplerClient

from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler


class AnnealingSolver(ABC):

    def __init__(self, token_file, proxy=None):
       self.proxy = proxy
       self.token = open(token_file,"r").read().replace('\n', '')
       self.setup_client()

    @abstractmethod
    def setup_client(self):
        pass

    @abstractmethod
    def setup_solver(self):
        pass

    @abstractmethod
    def solve_qubo_problem(self, problem):
        pass

class AnnealingSolverAmplify(AnnealingSolver):
    def __init__(self, client_type, token_file, proxy=None):
        self.client_type = client_type
        super().__init__(token_file, proxy)


    def setup_client(self):

        if self.client_type == 'fixstars':
            self.client = FixstarsClient(token=self.token)
        elif self.client_type == 'dwave':
            self.client = DWaveSamplerClient(token=self.token)
        elif self.client_type == 'dwave_hybrid':
            self.client = LeapHybridSamplerClient(token=self.token)
        else:
            raise Exception('Unknown client type', self.client_type)
        if self.proxy:
            self.client.proxy = self.proxy
        if self.client_type in ['dwave','dwave_hybrid']:
            print('Available solvers:', self.client.solver_names)

    
    def setup_solver(self):

        # Setting default parameters.
        if self.client_type == 'fixstars':# Fixstars
            print('Setting default timeout (ms): 800')
            self.client.parameters.timeout = 800
        elif self.client_type == 'dwave': # DWave
            print('Choosing default solver: Advantage_system4.1')
            self.client.solver = "Advantage_system4.1"
            print('Setting default num_reads: 200')
            self.client.parameters.num_reads = 200  # Number of executions
        elif self.client_type == 'dwave_hybrid': # DWave Hybrid
            print('Choosing default solver: hybrid_binary_quadratic_model_version2')
            self.client.solver = 'hybrid_binary_quadratic_model_version2'

        self.solver = Solver(self.client)
        print("Created solver")


    def solve_qubo_problem(self, problem):
        
        problem.results = self.solver.solve(problem.binary_quadratic_model)
        print('Number of solutions:', len(problem.results))

class AnnealingSolverDWave(AnnealingSolver):

    def __init__(self, token_file, proxy=None):
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
                self.solver = EmbeddingComposite(DWaveSampler(token=self.token, proxy=self.proxy, solver=dict(name=self.solver_name)))
            else:
                self.solver_type = 'hybrid'
                self.solver = LeapHybridSampler(token=self.token, proxy=self.proxy)

        
        elif solver_type:
            self.solver_type = solver_type
            if self.solver_type == 'qpu':
                self.solver = EmbeddingComposite(DWaveSampler(token=self.token, proxy=self.proxy))
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
            #problem.binary_quadratic_model.relabel_variables(problem.label_mapping_inverse,inplace=True)
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
