import numpy as np


class FuzzyProblem:
    def __init__(self, inputs):
        """
        inputs: dict
        example:
        {
            "error": 0.3,
            "delta_error": -0.1
        }
        """
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs