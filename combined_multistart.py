import numpy as np

from interval import interval
from typing import Callable

from global_optimization import hansen
from gradient_descent import GradientDescent
from lr_scheduler import HansenScheduler
from optimizer import experiment_wrapper


class CombinedMultistartOptimizer(GradientDescent):
    def __init__(
        self,
        func: Callable[..., float],
        func_grad: Callable[..., np.ndarray],
        func_bounder: Callable[..., Callable],
        *,
        num_args: int | None = None,
        tol: float = 1e-6,
        max_iter: int | None = None,
        grad_bounder: Callable[..., Callable] | None = None,
    ):
        super().__init__(HansenScheduler(), func, func_grad, func_bounder, num_args=num_args, tol=tol, max_iter=max_iter, grad_bounder=grad_bounder)

    @experiment_wrapper
    def optimize(self):
        initial_args = self.args.copy()
        iter_best = self.max_iter
        best_err = np.inf

        grad = self.func_grad(initial_args)
        L = hansen(self.func_bounder(initial_args, grad), interval[1e-3, 100], self.grad_bounder(initial_args, grad))[1]
        for X, _ in L:
            lr = (X[0].inf + X[0].sup) / 2
            self.args = initial_args - lr * grad
            self._iter_count = 0
            self._error = 0
            self.optimize_impl()
            if self.error < best_err:
                best_err = self.error
                iter_best = self.iter_count

        self._error = best_err
        self._iter_count = iter_best + 1
