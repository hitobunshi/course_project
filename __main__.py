import func
import lr_scheduler
import numpy as np

from argparse import ArgumentParser
from enum import Enum
from typing import Callable

from gradient_descent import GradientDescent


class AvailableFunctions(str, Enum):
    QUAD = 'quad'
    RASTRIGIN = 'rastrigin'
    ACKLEY = 'ackley'
    HIMMELBLAU = 'himmelblau'
    ROSENBROCK = 'rosenbrock'


class Scheduler(str, Enum):
    CONST_LR = 'ConstLrScheduler'
    HANSEN = 'HansenScheduler'


schedulers = {
    Scheduler.CONST_LR: lr_scheduler.ConstLrScheduler(1e-4),
    Scheduler.HANSEN: lr_scheduler.HansenScheduler(),
}


if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Function Optimizer',
        usage='Use this module to find optima of some pre-defined functions',
    )
    parser.add_argument('--function', type=AvailableFunctions, default=AvailableFunctions.QUAD, help='Function to optimize')
    parser.add_argument('--num-args', type=int, default=2, help='Number of arguments to pass to the function')
    parser.add_argument('--tol', type=float, default=1e-8, help='Tolerance for the optimization')
    parser.add_argument('--maximize', type=bool, default=False, help='Wether to maximize the function')
    parser.add_argument('--scheduler', type=Scheduler, default=Scheduler.HANSEN, help='LR scheduler to use')
    parser.add_argument('--max-iter', type=int, default=1e4, help='Maximum number of iterations in gradient descent')
    args = parser.parse_args()

    scheduler: lr_scheduler.LrScheduler = schedulers[args.scheduler]

    function: Callable[[np.ndarray], float] = getattr(func, args.function)
    grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
    bounder: Callable[[np.ndarray], float] = getattr(func, args.function + '_bounder')
    grad_bounder: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad_bounder', None)
    if args.maximize:
        old_grad: Callable[[np.ndarray], float] = getattr(func, args.function + '_grad')
        grad = lambda x: -1 * old_grad(x)

    gradient_descent = GradientDescent(scheduler, function, grad, bounder, num_args=args.num_args, tol=args.tol, max_iter=args.max_iter, grad_bounder=grad_bounder)
    min_point: np.ndarray = gradient_descent.descent()

    optimum_name = 'maximum' if args.maximize else 'minimum'
    print(f'Found {optimum_name}: {min_point}')
    print(f'Function value in the {optimum_name}: {function(min_point)}')
