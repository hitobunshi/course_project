import abc
import numpy as np

from functools import wraps
from time import perf_counter
from typing import Callable


def experiment_wrapper(f: Callable):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        self._iter_count = 0
        self._sum_error = 0
        self._time_start = perf_counter()
        f(self, *args, **kwargs)
        self._time_end = perf_counter()
    return wrapper


class Optimizer(abc.ABC):
    def __init__(self, func: Callable, func_grad: Callable, num_args: int, max_iter: int | None = None):
        super().__init__()
        self.func = func
        self.func_grad = func_grad

        self._iter_count: int = 0
        self._sum_error: float = 0
        self._time_start: float = 0
        self._time_end: float = 0

        self.max_iter = max_iter

        self.num_args = num_args
        self.args = np.random.rand(num_args)

    @abc.abstractmethod
    def optimize(self) -> None:
        pass

    @property
    def error(self) -> float:
        return self._sum_error

    @property
    def iter_count(self) -> int:
        return self._iter_count

    @property
    def time_sec(self) -> float:
        return self._time_end - self._time_start
