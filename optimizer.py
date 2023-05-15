import abc
import numpy as np

from typing import Callable


class Optimizer(abc.ABC):
    def __init__(self, func: Callable, func_grad: Callable, initial_args: np.ndarray | None = None, num_args: int | None = None, max_iter: int | None = None):
        super().__init__()
        self.func = func
        self.func_grad = func_grad

        self._iter_count: int = 0
        self._sum_error: float = 0
        self._time_start: float = 0
        self._time_end: float = 0

        self.max_iter = max_iter

        if initial_args:
            self.args = initial_args
            self.num_args = len(initial_args)
        elif num_args:
            self.num_args = num_args
            self.args = np.random.rand(num_args)
        else:
            raise ValueError("Either initial_args or num_args should be provided")

    @abc.abstractmethod
    def optimize(self) -> None:
        pass

    @property
    def mean_error(self) -> float:
        return self._sum_error / self._iter_count
    
    @property
    def iter_count(self) -> int:
        return self._iter_count
    
    @property
    def time_sec(self) -> float:
        return self._time_end - self._time_start
