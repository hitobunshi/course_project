import numpy as np

from abc import ABC, abstractmethod
from typing import Callable
from interval import interval, inf as iinf

from global_optimization import hansen


class LrScheduler(ABC):
    """Политика изменения длины шага"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(**kwargs) -> int:
        pass


class ConstLrScheduler(LrScheduler):
    """Костантная длина шага"""

    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

    def step(self, **kwargs) -> float:
        return self.lr


class HansenScheduler(LrScheduler):
    """Глобальная оптимизация на направлении антиградиента, шаг в точку глобального минимума одномерной проекции"""

    def __init__(self):
        pass

    def step(self, **kwargs) -> float:
        grad_bounder = kwargs['grad_bounder'] if 'grad_bounder' in kwargs else None
        box, _ = hansen(kwargs['bounder'], interval[1e-3, 100], F_grad=kwargs['grad_bounder'])[0]
        return (box[0].sup + box[0].inf) / 2
