class ResultInterpreter:
    def interpret(self, result: dict):
        fitness = result["best_fitness"]
        time = result["execution_time"]

        if fitness < 1e-3:
            quality = "GOOD"
        elif fitness < 1e-1:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"

        recommendations = []
        if quality != "GOOD":
            recommendations.append("Increase iterations or population size.")
            recommendations.append("Try alternative method for comparison.")

        return {
            "performance": quality,
            "execution_time": time,
            "recommendations": recommendations
        }