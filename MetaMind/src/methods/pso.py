import numpy as np
from .base_method import BaseMethod


class ParticleSwarmOptimization(BaseMethod):
    def initialize(self, problem):
        self.problem = problem

        self.n_particles = self.params.get("n_particles", 50)
        self.max_iterations = self.params.get("max_iterations", 300)
        self.w = self.params.get("w", 0.7)
        self.c1 = self.params.get("c1", 1.5)
        self.c2 = self.params.get("c2", 1.5)

        self.dim = problem.dimension
        self.lb = problem.lower_bound
        self.ub = problem.upper_bound

        # Initialize particles
        self.positions = np.random.uniform(self.lb, self.ub,
                                            (self.n_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1,
                                              (self.n_particles, self.dim))

        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array(
            [self.problem.evaluate(p) for p in self.positions]
        )

        best_idx = np.argmin(self.personal_best_fitness)
        self.best_solution = self.personal_best_positions[best_idx].copy()
        self.best_fitness = self.personal_best_fitness[best_idx]

    def run(self):
        self._start_timer()

        for it in range(self.max_iterations):
            for i in range(self.n_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.best_solution - self.positions[i])
                )

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                fitness = self.problem.evaluate(self.positions[i])

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()

                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = self.positions[i].copy()

            self.history.append(self.best_fitness)

        self._stop_timer()
