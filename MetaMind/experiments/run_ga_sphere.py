from src.methods.ga import GeneticAlgorithm
from src.problems.sphere import SphereProblem


problem = SphereProblem(dimension=10)

params = {
    "population_size": 60,
    "generations": 300,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1
}

ga = GeneticAlgorithm(params)
ga.initialize(problem)
ga.run()

results = ga.get_results()

print("Best fitness:", results["best_fitness"])
print("Execution time:", results["execution_time"])
