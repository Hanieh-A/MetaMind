import numpy as np
import time


class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = 0.0
        self.history = []

    def initialize(self, problem):
        self.X = problem.X
        self.y = problem.y
        n_features = self.X.shape[1]
        self.weights = np.zeros(n_features)

    def _activation(self, x):
        return 1 if x >= 0 else 0

    def run(self):
        start_time = time.time()

        for epoch in range(self.max_epochs):
            errors = 0

            for xi, target in zip(self.X, self.y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self._activation(linear_output)

                update = self.learning_rate * (target - prediction)

                if update != 0:
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.history.append(errors)

            if errors == 0:
                break

        self.execution_time = time.time() - start_time
        self.epochs_run = epoch + 1

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self._activation(x) for x in linear_output])

    def get_results(self):
        return {
            "best_fitness": self.history[-1],
            "epochs": self.epochs_run,
            "execution_time": self.execution_time
        }
