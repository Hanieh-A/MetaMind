import numpy as np


class TSPProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = np.array(distance_matrix)
        self.n_cities = self.distance_matrix.shape[0]

    def evaluate(self, tour):
        total_distance = 0.0
        for i in range(len(tour)):
            total_distance += self.distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return total_distance
