class MethodSelector:
    def select(self, problem_description: dict):
        problem_type = problem_description["problem_type"]

        if problem_type == "function_optimization":
            return {
                "selected_method": "PSO",
                "parameters": {
                    "n_particles": 50,
                    "max_iterations": 500,
                    "w": 0.7,
                    "c1": 1.5,
                    "c2": 1.5
                },
                "reasoning": "PSO performs well on continuous multimodal functions."
            }

        if problem_type == "tsp":
            return {
                "selected_method": "ACO",
                "parameters": {
                    "n_ants": 30,
                    "max_iterations": 500,
                    "alpha": 1.0,
                    "beta": 2.0,
                    "evaporation_rate": 0.5
                },
                "reasoning": "ACO is naturally suited for graph-based routing problems."
            }

        if problem_type == "classification":
            return {
                "selected_method": "MLP",
                "parameters": {
                    "hidden_layers": [64, 32],
                    "learning_rate": 0.01,
                    "max_epochs": 500
                },
                "reasoning": "MLP handles nonlinear decision boundaries effectively."
            }

        if problem_type == "clustering":
            return {
                "selected_method": "SOM",
                "parameters": {
                    "map_size": (10, 10),
                    "learning_rate_initial": 0.5,
                    "max_epochs": 1000
                },
                "reasoning": "SOM is effective for topology-preserving clustering."
            }

        raise ValueError("Unknown problem type")