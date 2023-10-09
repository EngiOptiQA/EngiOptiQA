from amplify import Solver
from amplify.client import FixstarsClient
from amplify.client.ocean import DWaveSamplerClient, LeapHybridSamplerClient

class AnnealingSolver():
    def __init__(self, client_type, token_file, proxy=None):
        self.client_type = client_type
        self.client = self.setup_client(token_file, proxy=proxy)
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

    def setup_client(self, token_file, proxy=None):

        token = open(token_file,"r").read().replace('\n', '')
        if self.client_type == 'fixstars':
            client = FixstarsClient(token=token)
        elif self.client_type == 'dwave':
            client = DWaveSamplerClient(token=token)
        elif self.client_type == 'dwave_hybrid':
            client = LeapHybridSamplerClient(token=token)
        else:
            raise Exception('Unknown client type', self.client_type)
        if proxy:
            client.proxy = proxy
        if self.client_type in ['dwave','dwave_hybrid']:
            print('Available solvers:', client.solver_names)

        return client

    def solve_QUBO_problem(self, problem):
        solver = Solver(self.client)
        problem.results = solver.solve(problem.binary_quadratic_model)
        print('Number of solutions:', len(problem.results))
