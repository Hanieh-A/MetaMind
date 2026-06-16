import numpy as np


class SphereProblem:
    def __init__(self, dimension=10, lower_bound=-5.12, upper_bound=5.12):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, x):
        return np.sum(x ** 2)

    def sample_solution(self):
        return np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            self.dimension
        )
