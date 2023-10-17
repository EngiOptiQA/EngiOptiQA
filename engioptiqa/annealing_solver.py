from abc import ABC, abstractmethod

from amplify import Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient, LeapHybridSamplerClient

from dwave.cloud import Client
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler


class AnnealingSolver(ABC):

    def __init__(self, token_file, proxy=None):
       self.proxy = proxy
       self.token = open(token_file,"r").read().replace('\n', '')
       self.client = self.setup_client()
       self.solver = self.setup_solver()

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
            client = FixstarsClient(token=self.token)
        elif self.client_type == 'dwave':
            client = DWaveSamplerClient(token=self.token)
        elif self.client_type == 'dwave_hybrid':
            client = LeapHybridSamplerClient(token=self.token)
        else:
            raise Exception('Unknown client type', self.client_type)
        if self.proxy:
            client.proxy = self.proxy
        if self.client_type in ['dwave','dwave_hybrid']:
            print('Available solvers:', client.solver_names)

        return client
    
    def setup_solver(self):

        # Setting default parameters.
        if self.client_type == 'fixstars':# Fixstars
            print('Setting default timeout (ms): 800')
            self.client.parameters.timeout = 800
        elif self.client_type == 'dwave': # DWave
            print('Choosing default solver: Advantage_system6.2')
            self.client.solver = "Advantage_system6.2"
            print('Setting default num_reads: 200')
            self.client.parameters.num_reads = 200  # Number of executions
        elif self.client_type == 'dwave_hybrid': # DWave Hybrid
            print('Choosing default solver: hybrid_binary_quadratic_model_version2')
            self.client.solver = 'hybrid_binary_quadratic_model_version2'

        solver = Solver(self.client)
        print("Created solver")

        return solver


    def solve_qubo_problem(self, problem):
        
        problem.results = self.solver.solve(problem.binary_quadratic_model)
        print('Number of solutions:', len(problem.results))

class AnnealingSolverDWave(AnnealingSolver):

    def __init__(self, solver_type, token_file, proxy=None):
        self.solver_type = solver_type
        super().__init__(token_file, proxy)

    def setup_client(self):

        self.client = Client(token=self.token, proxy=self.proxy)
        print("Available QPU solvers:")
        for solver in self.client.get_solvers(qpu=True, online=True):
            print("\t", solver)
        print("Available hybrid solvers:")
        for solver in self.client.get_solvers(name__regex='hybrid_binary.*', online=True):
            print("\t", solver)

    def setup_solver(self):

        if self.solver_type == 'quantum_annealing':
            if self.proxy:
                solver = EmbeddingComposite(DWaveSampler(token=self.token, proxy=self.proxy))
            else:
                solver = EmbeddingComposite(DWaveSampler(token=self.token))
            print("Created solver as EmbeddingComposite(DWaveSampler)")   
            print("Solver: ", solver.child.properties["chip_id"]) 

        elif self.solver_type == 'hybrid':
            if self.proxy:
                solver = LeapHybridSampler(token=self.token, proxy=self.proxy) 
            else:
                solver = LeapHybridSampler(token=self.token) 
            print("Created solver as LeapHybridSampler")
            print("Version: ", solver.child.properties["version"])  

        elif self.solver_type == 'simulated_annealing':
            solver = SimulatedAnnealingSampler() 
            print("Created solver as SimulatedAnnealingSampler")

        else:
            raise Exception('Unknown solver type', self.solver_type)

        return solver

    def solve_qubo_problem(self, problem, **kwargs):

        problem.results_indices = self.solver.sample(
            problem.binary_quadratic_model_indices, 
            **kwargs
        )
        print('Number of solutions:', len(problem.results_indices))
        problem.results =  problem.results_indices.relabel_variables(
            problem.label_mapping_inverse,
            inplace=False)
        #problem.binary_quadratic_model.relabel_variables(problem.label_mapping_inverse,inplace=True)

