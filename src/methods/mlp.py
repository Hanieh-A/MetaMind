import numpy as np
from .base_method import BaseMethod


class MLP(BaseMethod):
    def initialize(self, problem):
        self.problem = problem
        self.X, self.y = problem.get_data()

        # Parameters
        self.hidden_layers = self.params.get("hidden_layers", [32, 16])
        self.learning_rate = self.params.get("learning_rate", 0.01)
        self.max_epochs = self.params.get("max_epochs", 500)

        self.input_size = self.X.shape[1]
        self.output_size = len(np.unique(self.y))

        self.layers = [self.input_size] + self.hidden_layers + [self.output_size]

        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            self.weights.append(
                np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1
            )
            self.biases.append(np.zeros((1, self.layers[i + 1])))

        self.y_onehot = self._one_hot(self.y)

    def _one_hot(self, y):
        onehot = np.zeros((len(y), self.output_size))
        for i, label in enumerate(y):
            onehot[i, label] = 1
        return onehot

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def run(self):
        self._start_timer()

        for epoch in range(self.max_epochs):
            # ---------- Forward ----------
            activations = [self.X]
            zs = []

            for i in range(len(self.weights) - 1):
                z = activations[-1] @ self.weights[i] + self.biases[i]
                zs.append(z)
                activations.append(self._relu(z))

            z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
            zs.append(z_out)
            y_pred = self._softmax(z_out)
            activations.append(y_pred)

            # ---------- Loss ----------
            loss = -np.mean(
                np.sum(self.y_onehot * np.log(y_pred + 1e-9), axis=1)
            )
            self.history.append(loss)

            # ---------- Backward ----------
            delta = y_pred - self.y_onehot

            for i in reversed(range(len(self.weights))):
                dW = activations[i].T @ delta / len(self.X)
                dB = np.mean(delta, axis=0, keepdims=True)

                self.weights[i] -= self.learning_rate * dW
                self.biases[i] -= self.learning_rate * dB

                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._relu_derivative(zs[i - 1])

        # ---------- Final Evaluation ----------
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == self.y)

        self.best_solution = self.weights
        self.best_fitness = 1 - accuracy

        self._stop_timer()