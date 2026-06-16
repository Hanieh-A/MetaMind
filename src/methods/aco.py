import numpy as np
from .base_method import BaseMethod


class AntColonyOptimization(BaseMethod):
    def initialize(self, problem):
        self.problem = problem

        self.n_ants = self.params.get("n_ants", 30)
        self.max_iterations = self.params.get("max_iterations", 200)
        self.alpha = self.params.get("alpha", 1.0)
        self.beta = self.params.get("beta", 2.0)
        self.evaporation_rate = self.params.get("evaporation_rate", 0.5)
        self.q = self.params.get("q", 1.0)

        self.n_cities = problem.n_cities
        self.distances = problem.distance_matrix

        self.pheromone = np.ones((self.n_cities, self.n_cities))
        self.heuristic = 1 / (self.distances + 1e-10)

        self.best_fitness = float("inf")
        self.best_solution = None

    def _construct_solution(self):
        tour = [np.random.randint(self.n_cities)]

        while len(tour) < self.n_cities:
            current_city = tour[-1]
            probabilities = []

            for city in range(self.n_cities):
                if city not in tour:
                    prob = (
                        self.pheromone[current_city, city] ** self.alpha
                        * self.heuristic[current_city, city] ** self.beta
                    )
                else:
                    prob = 0
                probabilities.append(prob)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_city = np.random.choice(range(self.n_cities), p=probabilities)
            tour.append(next_city)

        return tour

    def _update_pheromones(self, tours, fitnesses):
        self.pheromone *= (1 - self.evaporation_rate)

        for tour, fitness in zip(tours, fitnesses):
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                self.pheromone[a, b] += self.q / fitness
                self.pheromone[b, a] += self.q / fitness

    def run(self):
        self._start_timer()

        for it in range(self.max_iterations):
            all_tours = []
            all_fitness = []

            for _ in range(self.n_ants):
                tour = self._construct_solution()
                fitness = self.problem.evaluate(tour)

                all_tours.append(tour)
                all_fitness.append(fitness)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = tour.copy()

            self._update_pheromones(all_tours, all_fitness)
            self.history.append(self.best_fitness)

        self._stop_timer()
