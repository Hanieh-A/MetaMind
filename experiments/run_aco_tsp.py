import numpy as np
from src.methods.aco import AntColonyOptimization
from src.problems.tsp import TSPProblem


np.random.seed(42)

n_cities = 10
coords = np.random.rand(n_cities, 2)

distance_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

problem = TSPProblem(distance_matrix)

params = {
    "n_ants": 20,
    "max_iterations": 200,
    "alpha": 1.0,
    "beta": 2.0,
    "evaporation_rate": 0.5,
}

aco = AntColonyOptimization(params)
aco.initialize(problem)
aco.run()

results = aco.get_results()

print("Best tour length:", results["best_fitness"])
print("Best tour:", results["best_solution"])
print("Execution time:", results["execution_time"])
