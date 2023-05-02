import numpy as np


def mean_quad(*args) -> float:
    return np.mean(np.square(np.array(args)))

def mean_quad_grad(*args) -> np.ndarray:
    return np.mean(np.array(args)) * 2
