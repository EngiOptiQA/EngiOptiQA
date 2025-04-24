from abc import ABC, abstractmethod

class AnnealingSolver(ABC):

    def __init__(self, token_file=None, proxy=None):
        self.proxy = proxy
        if token_file is not None:
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