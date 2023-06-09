import func
import lr_scheduler
import numpy as np

from argparse import ArgumentParser
from enum import Enum
from typing import Callable

from adam import AdamOptimizer
from bfgs import BFGSOptimizer
from gradient_descent import GradientDescent
from lr_scheduler import ConstLrScheduler, HansenScheduler
from combined_multistart import CombinedMultistartOptimizer


class AvailableFunctions(str, Enum):
    QUAD = 'quad'
    RASTRIGIN = 'rastrigin'
    ACKLEY = 'ackley'
    HIMMELBLAU = 'himmelblau'
    ROSENBROCK = 'rosenbrock'


class Method(str, Enum):
    GRADIENT_DESCENT = 'gradient_descent'
    COMBINED = 'combined'
    COMBINED_MULTISTART = 'combined_multistart'
    BFGS = 'bfgs'
    ADAM = 'adam'


if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Function Optimizer',
        usage='Use this module to find optima of some pre-defined functions',
    )
    parser.add_argument('--function', type=AvailableFunctions, default=AvailableFunctions.QUAD, help='Function to optimize')
    parser.add_argument('--num-args', type=int, default=2, help='Number of arguments to pass to the function')
    parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for the optimization')
    parser.add_argument('--maximize', type=bool, default=False, help='Wether to maximize the function')
    parser.add_argument('--method', type=Method, default=Method.GRADIENT_DESCENT, help='LR scheduler to use')
    parser.add_argument('--max-iter', type=int, default=None, help='Maximum number of iterations in gradient descent')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of experiment repetitions')
    args = parser.parse_args()

    function: Callable[[np.ndarray], float] = getattr(func, args.function)
    grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
    bounder: Callable[[np.ndarray], float] = getattr(func, args.function + '_bounder')
    grad_bounder: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad_bounder', None)
    if args.maximize:
        old_grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
        grad = lambda x: -1 * old_grad(x)

    match args.method:
        case Method.GRADIENT_DESCENT:
            optimizer = GradientDescent(ConstLrScheduler(1e-3), function, grad, bounder, num_args=args.num_args, tol=args.tol, max_iter=args.max_iter)
        case Method.COMBINED:
            optimizer = GradientDescent(HansenScheduler(), function, grad, bounder, num_args=args.num_args, tol=args.tol, max_iter=args.max_iter, grad_bounder=grad_bounder)
        case Method.COMBINED_MULTISTART:
            optimizer = CombinedMultistartOptimizer(function, grad, bounder, num_args=args.num_args, tol=args.tol, max_iter=args.max_iter, grad_bounder=grad_bounder)
        case Method.BFGS:
            optimizer = BFGSOptimizer(function, grad, args.num_args, max_iter=args.max_iter, tol=args.tol)
        case Method.ADAM:
            optimizer = AdamOptimizer(function, grad, args.num_args, max_iter=args.max_iter, tol=args.tol)

    sum_error: float = 0
    sum_time: float = 0
    sum_iter_count: float = 0
    for _ in range(args.repetitions):
        optimizer.optimize()
        sum_error += optimizer.error
        sum_time += optimizer.time_sec
        sum_iter_count += optimizer.iter_count

    print(f'Mean error: {sum_error / args.repetitions}')
    print(f'Mean time: {sum_time / args.repetitions} s')
    print(f'Mean iter count: {sum_iter_count / args.repetitions}')
