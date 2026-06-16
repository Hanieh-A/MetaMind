import numpy as np
from .base_method import BaseMethod


class GeneticAlgorithm(BaseMethod):
    def initialize(self, problem):
        self.problem = problem
        self.pop_size = self.params.get("population_size", 50)
        self.generations = self.params.get("generations", 200)
        self.crossover_rate = self.params.get("crossover_rate", 0.8)
        self.mutation_rate = self.params.get("mutation_rate", 0.1)

        self.population = [
            problem.sample_solution()
            for _ in range(self.pop_size)
        ]

    def _fitness(self, individual):
        return self.problem.evaluate(individual)

    def _select(self):
        i, j = np.random.randint(0, self.pop_size, 2)
        return self.population[i] if self._fitness(self.population[i]) < self._fitness(self.population[j]) else self.population[j]

    def _crossover(self, p1, p2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(p1))
            return np.concatenate([p1[:point], p2[point:]])
        return p1.copy()

    def _mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        return individual

    def run(self):
        self._start_timer()

        for gen in range(self.generations):
            new_population = []

            for _ in range(self.pop_size):
                p1 = self._select()
                p2 = self._select()
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

                fitness = self._fitness(child)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = child.copy()

            self.population = new_population
            self.history.append(self.best_fitness)

        self._stop_timer()
