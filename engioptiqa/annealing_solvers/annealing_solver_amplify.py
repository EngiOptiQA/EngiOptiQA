from amplify import Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient, LeapHybridSamplerClient

from .annealing_solver import AnnealingSolver

class AnnealingSolverAmplify(AnnealingSolver):
    """
        Initialize the Annealing Solver that uses the Amplify SDK.

        :param client_type: The client type.
        :param token_file: The path of the file that includes the token for the Fixstars Amplify Annealing Engine.
    """
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

        if self.client_type == 'fixstars':
            print('Setting default timeout (ms): 800')
            self.client.parameters.timeout = 800
        elif self.client_type == 'dwave':
            print('Choosing default solver: Advantage_system4.1')
            self.client.solver = "Advantage_system4.1"
            print('Setting default num_reads: 200')
            self.client.parameters.num_reads = 200 
        elif self.client_type == 'dwave_hybrid':
            print('Choosing default solver: hybrid_binary_quadratic_model_version2')
            self.client.solver = 'hybrid_binary_quadratic_model_version2'

        self.solver = Solver(self.client)
        print("Created solver")

    def solve_qubo_problem(self, problem):
        
        problem.results = self.solver.solve(problem.binary_quadratic_model)
        print('Number of solutions:', len(problem.results))
