from functools import lru_cache
import numpy as np
import math


@lru_cache
def sqrt_primes():
    return np.sqrt(np.array([2] + list(filter(lambda k: (k % np.arange(3, 1 + int(math.sqrt(k)), 2)).all(), range(1, 10000 + 1, 2)))))


def mean_quad(args: np.ndarray) -> float:
    return np.mean(np.square(args))


def mean_quad_grad(args: np.ndarray) -> np.ndarray:
    return args * 2 / len(args)


def sin_cos(args: np.ndarray) -> float:
    coefs = sqrt_primes()[:2 * len(args)]
    coefs_sin = coefs[:len(args)]
    coefs_cos = coefs[len(args):]
    return np.sum(np.sin(args * coefs_sin) + np.cos(args * coefs_cos))

def sin_cos_grad(args: np.ndarray) -> float:
    coefs = sqrt_primes()[:2 * len(args)]
    coefs_sin = coefs[:len(args)]
    coefs_cos = coefs[len(args):]
    return np.sum(np.cos(args * coefs_sin) - np.sin(args * coefs_cos))
