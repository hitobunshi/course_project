from abc import ABC, abstractmethod
import numpy as np


class LrScheduler(ABC):
    """Политика изменения длины шага"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(*args) -> int:
        pass


class ConstLrScheduler(LrScheduler):
    """Костантная длина шага"""
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

    def step(self, *args) -> float:
        return self.lr
