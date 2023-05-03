import func
import numpy as np

from argparse import ArgumentParser
from enum import Enum
from typing import Callable

from gradient_descent import GradientDescent
from lr_scheduler import ConstLrScheduler


class AvailableFunctions(str, Enum):
    MEAN_QUAD = 'mean_quad'
    SIN_COS = 'sin_cos'


if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Function Optimizer',
        usage='Use this module to find optima of some pre-defined functions',
    )
    parser.add_argument('--function', type=AvailableFunctions, default=AvailableFunctions.SIN_COS, help='Function to optimize')
    parser.add_argument('--num-args', type=int, default=2, help='Number of arguments to pass to the function')
    parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for the optimization')
    parser.add_argument('--maximize', type=bool, default=False, help='Wether to maximize the function')
    args = parser.parse_args()

    lr_scheduler = ConstLrScheduler(1e-4)

    function: Callable[[np.ndarray], float] = getattr(func, args.function)
    grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
    if args.maximize:
        old_grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
        grad = lambda x: -1 * old_grad(x)

    gradient_descent = GradientDescent(lr_scheduler, function, grad, num_args=args.num_args, tol=args.tol)
    min_point: np.ndarray = gradient_descent.descent()

    optimum_name = 'maximum' if args.maximize else 'minimum'
    print(f'Found {optimum_name}: {min_point}')
    print(f'Function value in the {optimum_name}: {function(min_point)}')
