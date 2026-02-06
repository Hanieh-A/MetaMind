import numpy as np


class HopfieldProblem:
    def __init__(self, patterns):
        """
        patterns: array-like, shape (n_patterns, n_neurons)
        values should be in {-1, +1}
        """
        self.patterns = np.array(patterns)
        self.n_patterns, self.n_neurons = self.patterns.shape

    def get_patterns(self):
        return self.patterns