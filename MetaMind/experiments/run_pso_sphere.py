from src.methods.pso import ParticleSwarmOptimization
from src.problems.sphere import SphereProblem


problem = SphereProblem(dimension=10)

params = {
    "n_particles": 40,
    "max_iterations": 300,
    "w": 0.7,
    "c1": 1.5,
    "c2": 1.5
}

pso = ParticleSwarmOptimization(params)
pso.initialize(problem)
pso.run()

results = pso.get_results()

print("Best fitness (PSO):", results["best_fitness"])
print("Execution time:", results["execution_time"])
