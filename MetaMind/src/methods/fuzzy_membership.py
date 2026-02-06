import numpy as np


def triangular(x, a, b, c):
    return max(min((x - a) / (b - a), (c - x) / (c - b)), 0)


def gaussian(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))