from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from src.methods.som import SelfOrganizingMap
from src.problems.clustering import ClusteringProblem


# Load data
iris = load_iris()
X = iris.data

# Normalize
X = MinMaxScaler().fit_transform(X)

problem = ClusteringProblem(X)

params = {
    "map_size": (8, 8),
    "learning_rate_initial": 0.5,
    "learning_rate_final": 0.01,
    "neighborhood_initial": 4.0,
    "max_epochs": 500
}

som = SelfOrganizingMap(params)
som.initialize(problem)
som.run()

results = som.get_results()
labels = som.assign_clusters()

print("Final quantization error:", results["best_fitness"])
print("Execution time:", results["execution_time"])
print("First 10 BMU labels:", labels[:10])