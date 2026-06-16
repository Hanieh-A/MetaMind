import numpy as np
from src.methods.perceptron import Perceptron
from src.problems.classification import ClassificationProblem

# Simple linearly separable dataset
X = np.array([
    [2, 3],
    [1, 1],
    [-1, -1],
    [-2, -3]
])

y = np.array([1, 1, 0, 0])

problem = ClassificationProblem(X, y)

model = Perceptron(
    learning_rate=0.1,
    max_epochs=100
)

model.initialize(problem)
model.run()

predictions = model.predict(X)

print("Predictions:", predictions)
print("Results:", model.get_results())
