import numpy as np

from typing import Callable

from lr_scheduler import ExponentialLrScheduler
from optimizer import experiment_wrapper, Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, func: Callable, func_grad: Callable, num_args: int, max_iter: int | None = None, tol: float = 1e-6):
        super().__init__(func, func_grad, num_args, max_iter=max_iter)
        self.tol = tol
        self.lr_scheduler = ExponentialLrScheduler(lr=1, gamma=0.9)
        self.beta = [0.9, 0.999]
        self.eps = 1e-8

    @experiment_wrapper
    def optimize(self) -> None:
        m = 0
        v = 0
        grad = np.inf
        self.lr_scheduler = ExponentialLrScheduler()

        while np.linalg.norm(grad) > self.tol and (self.max_iter is None or self._iter_count < self.max_iter):
            self._iter_count += 1
            grad = self.func_grad(self.args)
            m = self.beta[0] * m + (1 - self.beta[0]) * grad
            v = self.beta[1] * v + (1 - self.beta[1]) * np.square(grad)
            m_norm = m / (1 - self.beta[0] ** self._iter_count)
            v_norm = v / (1 - self.beta[1] ** self._iter_count)
            self.args -= self.lr_scheduler.step() * m_norm / (np.sqrt(v_norm) + self.eps)

        self._error = self.func(self.args)
