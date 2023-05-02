import numpy as np

from typing import Callable

from lr_scheduler import LrScheduler


class GradientDescent:
    """Базовый градиентный спуск"""
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        func: Callable[..., float],
        func_grad: Callable[..., np.ndarray],
        *,
        num_args: int | None = None,
        initial_args: np.ndarray | None = None,
        tol: float = 1e-6,
        max_iter: int | None = None,
    ):
        self.lr_scheduler = lr_scheduler
        self.tol = tol
        self.func = func
        self.func_grad = func_grad
        self.max_iter = max_iter

        if initial_args:
            self.args = initial_args
            self.num_args = len(initial_args)
        elif num_args:
            self.num_args = num_args
            self.args = np.random.rand(num_args)
        else:
            raise ValueError("Either initial_args or num_args should be provided")
        
    def descent(self) -> np.ndarray:
        """Проводит градиентный спуск и возвращает точку оптимума"""
        grad: float = np.inf
        it: int = 0

        while np.linalg.norm(grad) > self.tol and (self.max_iter is None or it < self.max_iter):
            grad = self.func_grad(*self.args)
            self.args -= self.lr_scheduler.step(self.args) * grad
            it += 1

        return self.args
