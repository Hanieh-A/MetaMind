from .method_selector import MethodSelector
from .result_interpreter import ResultInterpreter


class LLMOrchestrator:
    def __init__(self):
        self.selector = MethodSelector()
        self.interpreter = ResultInterpreter()

    def analyze_problem(self, problem_description: dict):
        decision = self.selector.select(problem_description)

        return {
            "problem_type": problem_description["problem_type"],
            "selected_method": decision["selected_method"],
            "parameters": decision["parameters"],
            "reasoning": decision["reasoning"],
            "confidence": 0.85
        }

    def interpret_results(self, results: dict):
        return self.interpreter.interpret(results)