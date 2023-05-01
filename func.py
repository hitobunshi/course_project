import numpy as np


def mean_quad(*args) -> float:
    return np.sum(np.square(np.array(args))) / len(args)

def mean_quad_grad(*args) -> np.ndarray:
    return np.array(args) * 2 / len(args)
