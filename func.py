from functools import lru_cache
import numpy as np
import math


@lru_cache
def sqrt_primes():
    return np.sqrt(np.array([2] + list(filter(lambda k: (k % np.arange(3, 1 + int(math.sqrt(k)), 2)).all(), range(1, 10000 + 1, 2)))))


def mean_quad(*args) -> float:
    return np.mean(np.square(np.array(args)))


def mean_quad_grad(*args) -> np.ndarray:
    return np.array(args) * 2 / len(args)


def sin_cos(*args) -> float:
    arr = np.array(args)
    coefs = sqrt_primes()[:2 * len(args)]
    coefs_sin = coefs[:len(args)]
    coefs_cos = coefs[len(args):]
    return np.sum(np.sin(arr * coefs_sin) + np.cos(arr * coefs_cos))

def sin_cos_grad(*args) -> float:
    arr = np.array(args)
    coefs = sqrt_primes()[:2 * len(args)]
    coefs_sin = coefs[:len(args)]
    coefs_cos = coefs[len(args):]
    return np.sum(np.sin(arr * coefs_cos) - np.cos(arr * coefs_sin))
