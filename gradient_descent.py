import numpy as np

from typing import Callable

from lr_scheduler import LrScheduler, HansenScheduler
from optimizer import experiment_wrapper, Optimizer


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
        tol: float = 1e-6,
        max_iter: int | None = None,
        grad_bounder: Callable[..., Callable] | None = None,
    ):
        super().__init__(func, func_grad, num_args, max_iter)
        self.lr_scheduler = lr_scheduler
        self.tol = tol
        self.func_bounder = func_bounder
        self.grad_bounder = grad_bounder

    @experiment_wrapper
    def optimize(self) -> None:
        """Проводит градиентный спуск и возвращает точку оптимума"""
        grad: np.ndarray = None
        grad_norm: float = np.inf

        while grad_norm > self.tol and (self.max_iter is None or self._iter_count < self.max_iter):
            grad = self.func_grad(self.args)
            grad_norm = np.linalg.norm(grad)

            if isinstance(self.lr_scheduler, HansenScheduler):
                lr = self.lr_scheduler.step(bounder=self.func_bounder(self.args, grad), grad_bounder=self.grad_bounder(self.args, grad))
            else:
                lr = self.lr_scheduler.step()

            self.args -= lr * grad
            self._iter_count += 1

        self._sum_error = self.func(self.args)
