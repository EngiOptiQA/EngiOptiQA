import itertools
from types import SimpleNamespace

class BruteForceSolver:
    def __init__(self, max_vars: int = 20):
        self.max_vars = max_vars

    def objective_from_pubo(self, pubo, binary_solution):
        objective = pubo.get((), 0.0)  # Constant term
        for indices, coeff in pubo.items():
            if not indices:  # Skip constant, already added
                continue
            # Contribution iff all variables with indices are 1
            active = True
            for i in indices:
                if binary_solution[i] == 0:
                    active = False
                    break
            if active:
                objective += coeff
        return objective

    def sample_objective(self, problem):
        model = problem.binary_model
        num_vars = len(model.variables)
        if num_vars > self.max_vars:
            msg = f"Brute-force infeasible: num_vars={num_vars} leads to 2^{num_vars} evaluations."
            raise ValueError(msg + f" Increase max_vars (current {self.max_vars}) or reduce problem size.")

        best_solution = None
        best_objective_value = float('inf')
        solutions = []
        for binary_solution in itertools.product([0, 1], repeat=num_vars):
            objective_value = self.objective_from_pubo(model.objective.as_dict(), binary_solution)
            binary_solution_dict = {i: binary_solution[i] for i in range(num_vars)}
            solution = SimpleNamespace(values=binary_solution_dict, energy=objective_value, frequency=1)
            solutions.append(solution)


            if objective_value < best_objective_value:
                best_objective_value = objective_value
                best_solution = binary_solution

        best_solution_dict = {i: best_solution[i] for i in range(num_vars)}
        best_solution = SimpleNamespace(values=best_solution_dict, energy=best_objective_value, frequency=1)

        return solutions, best_solution

    def solve_problem(self, problem):
        solutions, best_solution = self.sample_objective(problem)
        problem.results = [best_solution]