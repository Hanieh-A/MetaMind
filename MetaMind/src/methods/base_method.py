from abc import ABC, abstractmethod
import time


class BaseMethod(ABC):
    def __init__(self, params: dict):
        self.params = params
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def initialize(self, problem):
        pass

    @abstractmethod
    def run(self):
        pass

    def get_results(self):
        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "history": self.history,
            "execution_time": self.end_time - self.start_time
        }

    def _start_timer(self):
        self.start_time = time.time()

    def _stop_timer(self):
        self.end_time = time.time()
