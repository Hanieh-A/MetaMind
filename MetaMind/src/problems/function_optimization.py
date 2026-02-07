import numpy as np


class FunctionOptimizationProblem:
    def __init__(self, func, dimension, bounds, minimize=True):
        """
        func: callable -> f(x)
        dimension: int
        bounds: tuple (lower, upper)
        minimize: bool
        """
        self.func = func
        self.dimension = dimension
        self.bounds = bounds
        self.minimize = minimize

    def evaluate(self, x):
        value = self.func(x)
        return value if self.minimize else -value

    def random_solution(self):
        low, high = self.bounds
        return np.random.uniform(low, high, self.dimension)