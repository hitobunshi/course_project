import numpy as np

from scipy.optimize import BFGS
from typing import Callable

from optimizer import experiment_wrapper, Optimizer


class BFGSOptimizer(Optimizer):
    def __init__(self, func: Callable, func_grad: Callable, num_args: int, max_iter: int | None = None, tol: float = 1e-6):
        super().__init__(func, func_grad, num_args, max_iter=max_iter)
        self.tol = tol

    @experiment_wrapper
    def optimize(self) -> None:
        J = BFGS()
        J.initialize(self.num_args, approx_type='inv_hess')
        grad = self.func_grad(self.args)

        while np.linalg.norm(grad) > self.tol and (self.max_iter is None or self._iter_count < self.max_iter):
            delta_x = -J.dot(self.func_grad(self.args))
            self.args += delta_x
            delta_grad = self.func_grad(self.args) - grad
            J.update(delta_x, delta_grad)
            grad += delta_grad
            self._iter_count += 1

        self._error = abs(self.func(self.args))
