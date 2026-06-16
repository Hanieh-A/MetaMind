import numpy as np
import random
import time


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value == "+":
            return self.left.evaluate(x) + self.right.evaluate(x)
        if self.value == "-":
            return self.left.evaluate(x) - self.right.evaluate(x)
        if self.value == "*":
            return self.left.evaluate(x) * self.right.evaluate(x)
        if self.value == "/":
            denom = self.right.evaluate(x)
            return self.left.evaluate(x) / denom if abs(denom) > 1e-6 else 1
        if self.value == "x":
            return x
        return float(self.value)

    def copy(self):
        return Node(
            self.value,
            self.left.copy() if self.left else None,
            self.right.copy() if self.right else None
        )


class GeneticProgramming:
    def __init__(
        self,
        population_size=50,
        max_depth=4,
        generations=100,
        mutation_rate=0.2
    ):
        self.population_size = population_size
        self.max_depth = max_depth
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.functions = ["+", "-", "*", "/"]
        self.terminals = ["x", -5, -3, -1, 1, 3, 5]

    def initialize(self, problem):
        self.x = problem.x
        self.y = problem.y
        self.population = [self._random_tree(self.max_depth)
                           for _ in range(self.population_size)]

    def _random_tree(self, depth):
        if depth == 0 or random.random() < 0.3:
            return Node(random.choice(self.terminals))
        func = random.choice(self.functions)
        return Node(
            func,
            self._random_tree(depth - 1),
            self._random_tree(depth - 1)
        )

    def _fitness(self, tree):
        error = 0
        for xi, yi in zip(self.x, self.y):
            try:
                pred = tree.evaluate(xi)
            except:
                pred = 0
            error += (pred - yi) ** 2
        return error / len(self.x)

    def _tournament(self, k=3):
        competitors = random.sample(self.population, k)
        return min(competitors, key=self._fitness)

    def _crossover(self, t1, t2):
        if random.random() < 0.5:
            return t1.copy()
        child = t1.copy()
        child.left = t2.copy()
        return child

    def _mutate(self, tree, depth):
        if random.random() < self.mutation_rate:
            return self._random_tree(depth)
        if tree.left:
            tree.left = self._mutate(tree.left, depth - 1)
        if tree.right:
            tree.right = self._mutate(tree.right, depth - 1)
        return tree

    def run(self):
        start = time.time()

        for _ in range(self.generations):
            new_population = []

            for _ in range(self.population_size):
                parent1 = self._tournament()
                parent2 = self._tournament()
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, self.max_depth)
                new_population.append(child)

            self.population = new_population

        self.best_tree = min(self.population, key=self._fitness)
        self.best_fitness = self._fitness(self.best_tree)
        self.execution_time = time.time() - start

    def get_results(self):
        return {
            "best_fitness": self.best_fitness,
            "execution_time": self.execution_time,
            "best_program": self.best_tree.value
        }
