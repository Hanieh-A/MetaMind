import numpy as np
from src.methods.hopfield import HopfieldNetwork
from src.problems.hopfield import HopfieldProblem


# Define binary patterns (-1, +1)
patterns = np.array([
    [1, -1, 1, -1, 1, -1, 1, -1],
    [1,  1, -1, -1, 1,  1, -1, -1],
    [-1, 1, -1, 1, -1, 1, -1, 1]
])

problem = HopfieldProblem(patterns)

params = {
    "max_iterations": 100,
    "async_update": True,
    "energy_threshold": 1e-6
}

hopfield = HopfieldNetwork(params)
hopfield.initialize(problem)
hopfield.run()

results = hopfield.get_results()

print("Recovered pattern:", results["best_solution"])
print("Final energy:", results["best_fitness"])
print("Execution time:", results["execution_time"])