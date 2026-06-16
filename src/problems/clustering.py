import numpy as np


class ClusteringProblem:
    def __init__(self, data):
        self.data = np.array(data)
        self.n_samples, self.n_features = self.data.shape

    def get_data(self):
        return self.data