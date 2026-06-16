import numpy as np
from src.methods.gp import GeneticProgramming
from src.problems.symbolic_regresion import SymbolicRegressionProblem


def target_function(x):
    return x**2 + x + 1


problem = SymbolicRegressionProblem(target_function)

gp = GeneticProgramming(
    population_size=60,
    generations=150,
    max_depth=4
)

gp.initialize(problem)
gp.run()

print("Best fitness:", gp.best_fitness)
print("Execution time:", gp.execution_time)