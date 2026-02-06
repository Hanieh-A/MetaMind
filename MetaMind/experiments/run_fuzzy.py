from src.methods.fuzzy import FuzzyController
from src.problems.fuzzy import FuzzyProblem


inputs = {
    "error": 0.3,
    "delta_error": -0.2
}

problem = FuzzyProblem(inputs)

params = {
    "membership_type": "triangular",
    "defuzzification": "centroid"
}

fuzzy = FuzzyController(params)
fuzzy.initialize(problem)
fuzzy.run()

results = fuzzy.get_results()

print("Fuzzy output:", results["best_solution"])
print("Execution time:", results["execution_time"])