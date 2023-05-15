import numpy as np

from time import perf_counter
from typing import Callable

from lr_scheduler import LrScheduler
from optimizer import Optimizer


class GradientDescent(Optimizer):
    """Базовый градиентный спуск"""
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        func: Callable[..., float],
        func_grad: Callable[..., np.ndarray],
        func_bounder: Callable[..., Callable],
        *,
        num_args: int | None = None,
        initial_args: np.ndarray | None = None,
        tol: float = 1e-6,
        max_iter: int | None = None,
        grad_bounder: Callable[..., Callable] | None = None,
    ):
        super().__init__(func, func_grad, initial_args, num_args, max_iter)
        self.lr_scheduler = lr_scheduler
        self.tol = tol
        self.func_bounder = func_bounder
        self.grad_bounder = grad_bounder

    def optimize(self) -> None:
        """Проводит градиентный спуск и возвращает точку оптимума"""
        self._time_start = perf_counter()

        grad: np.ndarray = None
        grad_norm: float = np.inf
        it: int = 0

        while grad_norm > self.tol and (self.max_iter is None or it < self.max_iter):
            grad = self.func_grad(self.args)
            grad_norm = np.linalg.norm(grad)
            lr = self.lr_scheduler.step(bounder=self.func_bounder(self.args, grad), grad_bounder=self.grad_bounder(self.args, grad))
            self.args -= lr * grad
            it += 1

        self._sum_error += abs(self.func(self.args))

        self._time_end = perf_counter()
