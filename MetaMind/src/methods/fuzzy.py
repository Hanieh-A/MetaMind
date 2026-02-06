import numpy as np
from .base_method import BaseMethod
from .fuzzy_membership import triangular


class FuzzyController(BaseMethod):
    def initialize(self, problem):
        self.problem = problem
        self.inputs = problem.get_inputs()

        # Parameters
        self.mf_type = self.params.get("membership_type", "triangular")
        self.defuzzification = self.params.get("defuzzification", "centroid")

        # Define universes
        self.universe = {
            "error": (-1, 1),
            "delta_error": (-1, 1),
            "output": (-1, 1)
        }

        # Membership functions
        self.mf = {
            "error": {
                "N": lambda x: triangular(x, -1, -1, 0),
                "Z": lambda x: triangular(x, -0.5, 0, 0.5),
                "P": lambda x: triangular(x, 0, 1, 1)
            },
            "delta_error": {
                "N": lambda x: triangular(x, -1, -1, 0),
                "Z": lambda x: triangular(x, -0.5, 0, 0.5),
                "P": lambda x: triangular(x, 0, 1, 1)
            },
            "output": {
                "N": lambda x: triangular(x, -1, -1, 0),
                "Z": lambda x: triangular(x, -0.5, 0, 0.5),
                "P": lambda x: triangular(x, 0, 1, 1)
            }
        }

        # Rule base (Wang–Mendel style)
        self.rules = [
            (("N", "N"), "N"),
            (("N", "Z"), "N"),
            (("N", "P"), "Z"),
            (("Z", "N"), "N"),
            (("Z", "Z"), "Z"),
            (("Z", "P"), "P"),
            (("P", "N"), "Z"),
            (("P", "Z"), "P"),
            (("P", "P"), "P"),
        ]

    def run(self):
        self._start_timer()

        error = self.inputs["error"]
        delta_error = self.inputs["delta_error"]

        # Fuzzification
        fuzz_error = {k: mf(error) for k, mf in self.mf["error"].items()}
        fuzz_delta = {k: mf(delta_error) for k, mf in self.mf["delta_error"].items()}

        # Inference
        output_activation = {"N": 0, "Z": 0, "P": 0}
        for (e_label, de_label), out_label in self.rules:
            strength = min(fuzz_error[e_label], fuzz_delta[de_label])
            output_activation[out_label] = max(output_activation[out_label], strength)

        # Defuzzification (centroid)
        xs = np.linspace(-1, 1, 200)
        num, den = 0.0, 0.0

        for x in xs:
            mu = max(
                min(output_activation["N"], self.mf["output"]["N"](x)),
                min(output_activation["Z"], self.mf["output"]["Z"](x)),
                min(output_activation["P"], self.mf["output"]["P"](x)),
            )
            num += x * mu
            den += mu

        output_value = num / den if den != 0 else 0.0

        self.best_solution = output_value
        self.best_fitness = abs(output_value)
        self.history.append(output_value)

        self._stop_timer()