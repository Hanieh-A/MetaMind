import numpy as np
from .base_method import BaseMethod


class HopfieldNetwork(BaseMethod):
    def initialize(self, problem):
        self.problem = problem
        self.patterns = problem.get_patterns()

        # Parameters
        self.max_iterations = self.params.get("max_iterations", 100)
        self.threshold = self.params.get("threshold", 0.0)
        self.async_update = self.params.get("async_update", True)
        self.energy_threshold = self.params.get("energy_threshold", 1e-6)

        self.n_neurons = self.patterns.shape[1]

        # Weight matrix (Hebbian learning)
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        for p in self.patterns:
            self.weights += np.outer(p, p)

        np.fill_diagonal(self.weights, 0)
        self.weights /= self.n_neurons

    def _energy(self, state):
        return -0.5 * state @ self.weights @ state + self.threshold * np.sum(state)

    def _update_async(self, state):
        indices = np.random.permutation(self.n_neurons)
        for i in indices:
            net = np.dot(self.weights[i], state) - self.threshold
            state[i] = 1 if net >= 0 else -1
        return state

    def _update_sync(self, state):
        net = self.weights @ state - self.threshold
        return np.where(net >= 0, 1, -1)

    def run(self):
        self._start_timer()

        # Start from noisy version of first pattern
        state = self.patterns[0].copy()
        noise = np.random.choice(
            [-1, 1], size=self.n_neurons, p=[0.1, 0.9]
        )
        state *= noise

        prev_energy = self._energy(state)

        for _ in range(self.max_iterations):
            if self.async_update:
                state = self._update_async(state)
            else:
                state = self._update_sync(state)

            energy = self._energy(state)
            self.history.append(energy)

            if abs(prev_energy - energy) < self.energy_threshold:
                break

            prev_energy = energy

        self.best_solution = state
        self.best_fitness = energy

        self._stop_timer()
