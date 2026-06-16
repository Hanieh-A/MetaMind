from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from src.methods.mlp import MLP
from src.problems.classification import ClassificationProblem


# Generate toy dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_classes=2,
    random_state=42
)

X = MinMaxScaler().fit_transform(X)

problem = ClassificationProblem(X, y)

params = {
    "hidden_layers": [32, 16],
    "learning_rate": 0.05,
    "max_epochs": 300
}

mlp = MLP(params)
mlp.initialize(problem)
mlp.run()

results = mlp.get_results()

print("Final error (1 - accuracy):", results["best_fitness"])
print("Execution time:", results["execution_time"])