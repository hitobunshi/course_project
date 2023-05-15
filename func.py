import numpy as np
import math

from interval import interval, imath
from typing import Callable


# ---------------------------------------------------


def quad(args: np.ndarray) -> float:
    return np.mean(np.square(args))


def quad_grad(args: np.ndarray) -> np.ndarray:
    return args * 2 / len(args)


def quad_bounder(coords: np.ndarray, grad: np.ndarray):
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        return sum([(interval[coord] + X * grad_coord) ** 2 for coord, grad_coord in zip(coords, antigrad)])
    
    return bounder


def quad_grad_bounder(coords: np.ndarray, grad: np.ndarray):
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        return sum([2 * grad_coord * (interval[coord] + X * grad_coord) for coord, grad_coord in zip(coords, antigrad)])
    
    return bounder


# ---------------------------------------------------


def rastrigin(args: np.ndarray) -> float:
    n = len(args)
    return 10 * n + np.sum(np.square(args) - 10 * np.cos(2 * np.pi * args))


def rastrigin_grad(args: np.ndarray) -> np.ndarray:
    return 2 * args + 20 * np.pi * np.sin(2 * np.pi * args)


def rastrigin_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах
    n = len(coords)

    def bounder(X: interval) -> interval:
        interval_coords: list[interval] = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        return interval[10 * n] + sum([coord ** 2 - 10 * imath.cos(2 * imath.pi * coord) for coord in interval_coords])
    
    return bounder


def rastrigin_grad_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        return sum([2 * grad_coord * (interval[coord] + X * grad_coord) + 20 * imath.pi * grad_coord * imath.sin(2 * imath.pi * (interval[coord] + X * grad_coord)) for coord, grad_coord in zip(coords, antigrad)])
    
    return bounder


# ---------------------------------------------------


def ackley(args: np.ndarray) -> float:
    x, y = args
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def ackley_grad(args: np.ndarray) -> np.ndarray:
    x, y = args
    return np.array([
        2 * x * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2)) + np.pi * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) * np.sin(2 * np.pi * x),
        2 * y * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) / np.sqrt(0.5 * (x ** 2 + y ** 2)) + np.pi * np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) * np.sin(2 * np.pi * y)
    ])


def ackley_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def isqrt(i: interval) -> interval:
        return interval[math.sqrt(i[0].inf), math.sqrt(i[0].sup)]

    def bounder(X: interval) -> interval:
        x, y = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        return -20 * imath.exp(-0.2 * isqrt(0.5 * (x ** 2 + y ** 2))) - imath.exp(0.5 * (imath.cos(2 * imath.pi * x) + imath.cos(2 * imath.pi * y))) + imath.e + 20
    
    return bounder


def ackley_grad_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def isqrt(i: interval) -> interval:
        return interval[math.sqrt(i[0].inf), math.sqrt(i[0].sup)]

    def bounder(X: interval) -> interval:
        x, y = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        grad_x, grad_y = antigrad
        return 2 * (grad_x * x + grad_y * y) * imath.exp(-0.2 * isqrt(0.5 * (x ** 2 + y ** 2))) / isqrt(0.5 * (x ** 2 + y ** 2)) + imath.pi * imath.exp(0.5 * (imath.cos(2 * imath.pi * x) + imath.cos(2 * imath.pi * y))) * (grad_x * imath.sin(2 * imath.pi * x) + grad_y * imath.sin(2 * imath.pi * y))
    
    return bounder

# ---------------------------------------------------


def himmelblau(args: np.ndarray) -> float:
    x, y = args
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def himmelblau_grad(args: np.ndarray) -> np.ndarray:
    x, y = args
    return np.array([
        2 * (x ** 2 + y - 11) * 2 * x + 2 * (x + y ** 2 - 7),
        2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * 2 * y,
    ])


def himmelblau_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        x, y = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    
    return bounder


def himmelblau_grad_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        x, y = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        grad_x, grad_y = antigrad
        return 2 * (x ** 2 + y - 11) * (2 * grad_x + grad_y) + 2 * (x + y ** 2 - 7) * (grad_x + 2 * grad_y)
    
    return bounder


# ---------------------------------------------------


def rosenbrock(args: np.ndarray) -> float:
    first = args[:-1]
    last = args[1:]
    return np.sum(np.square(1 - first) + 100 * np.square(last - np.square(first)))


def rosenbrock_grad(args: np.ndarray) -> np.ndarray:
    res = [-2 * (1 - args[0]) - 400 * args[0] * (args[1] - args[0] ** 2)]
    for i in range(1, len(args) - 1):
        res.append(-2 * (1 - args[i]) - 400 * args[i] * (args[i + 1] - args[i] ** 2) + 200 * (args[i] - args[i - 1] ** 2))
    res.append(200 * (args[-1] - args[-2] ** 2))
    return np.array(res)


def rosenbrock_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        interval_coords: list[interval] = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        return sum([(1 - interval_coords[i]) ** 2 + 100 * (interval_coords[i + 1] - interval_coords[i] ** 2) ** 2 for i in range(len(interval_coords) - 1)])

    return bounder


def rosenbrock_grad_bounder(coords: np.ndarray, grad: np.ndarray) -> Callable[[interval], interval]:
    antigrad = [interval[coord] for coord in -grad]  # антиградиент в интервалах

    def bounder(X: interval) -> interval:
        interval_coords: list[interval] = [interval[coord] + X * grad_coord for coord, grad_coord in zip(coords, antigrad)]
        return sum([-2 * (1 - interval_coords[i]) * antigrad[i] + 200 * (interval_coords[i + 1] - interval_coords[i] ** 2) * (antigrad[i + 1] - 2 * interval_coords[i] * antigrad[i]) for i in range(len(coords) - 1)])

    return bounder
