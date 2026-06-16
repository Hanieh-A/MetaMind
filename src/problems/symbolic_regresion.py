import numpy as np


class SymbolicRegressionProblem:
    def __init__(self, func, x_range=(-5, 5), n_samples=50):
        self.x = np.linspace(x_range[0], x_range[1], n_samples)
        self.y = func(self.x)