from src.orchestrator.llm_orchestrator import LLMOrchestrator
from src.methods.pso import ParticleSwarmOptimization
from src.problems.sphere import SphereProblem


# --- Problem description (LLM input) ---
problem_description = {
    "problem_type": "function_optimization",
    "dimension": 10,
    "priority": "solution_quality"
}

# --- Orchestrator ---
orchestrator = LLMOrchestrator()
decision = orchestrator.analyze_problem(problem_description)

print("LLM Decision:")
print(decision)

# --- Execute selected method ---
problem = SphereProblem(dimension=10)
method = ParticleSwarmOptimization(decision["parameters"])
method.initialize(problem)
method.run()

results = method.get_results()

# --- Interpret results ---
analysis = orchestrator.interpret_results(results)

print("\nLLM Analysis:")
print(analysis)