import numpy as np
from .base_method import BaseMethod


class SelfOrganizingMap(BaseMethod):
    def initialize(self, problem):
        self.problem = problem
        self.data = problem.get_data()

        # Parameters
        self.map_size = self.params.get("map_size", (10, 10))
        self.lr_initial = self.params.get("learning_rate_initial", 0.5)
        self.lr_final = self.params.get("learning_rate_final", 0.01)
        self.sigma_initial = self.params.get("neighborhood_initial", 5.0)
        self.max_epochs = self.params.get("max_epochs", 1000)

        self.rows, self.cols = self.map_size
        self.dim = self.data.shape[1]

        # Weight initialization
        self.weights = np.random.rand(self.rows, self.cols, self.dim)

    def _decay(self, initial, final, epoch):
        return initial * (final / initial) ** (epoch / self.max_epochs)

    def _find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _neighborhood(self, bmu, sigma):
        grid_x, grid_y = np.meshgrid(
            np.arange(self.rows), np.arange(self.cols), indexing="ij"
        )
        dist_sq = (grid_x - bmu[0]) ** 2 + (grid_y - bmu[1]) ** 2
        return np.exp(-dist_sq / (2 * sigma ** 2))

    def run(self):
        self._start_timer()

        for epoch in range(self.max_epochs):
            lr = self._decay(self.lr_initial, self.lr_final, epoch)
            sigma = self._decay(self.sigma_initial, 1.0, epoch)

            for sample in self.data:
                bmu = self._find_bmu(sample)
                h = self._neighborhood(bmu, sigma)

                self.weights += lr * h[:, :, np.newaxis] * (sample - self.weights)

            # Quantization error (for convergence monitoring)
            error = 0.0
            for sample in self.data:
                bmu = self._find_bmu(sample)
                error += np.linalg.norm(sample - self.weights[bmu])

            self.history.append(error / len(self.data))

        self.best_solution = self.weights
        self.best_fitness = self.history[-1]

        self._stop_timer()

    def assign_clusters(self):
        labels = []
        for sample in self.data:
            bmu = self._find_bmu(sample)
            labels.append(bmu)
        return labels