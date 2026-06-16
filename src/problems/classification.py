import numpy as np


class ClassificationProblem:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        self.n_samples, self.n_features = self.X.shape
        self.n_classes = len(np.unique(y))

    def get_data(self):
        return self.X, self.y