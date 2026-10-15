import numpy as np
from numpy.typing import NDArray

from pyml.base import Estimator


class Optimizer:
    losses: NDArray

    def run(self, estimator: Estimator, x: NDArray, y: NDArray, params: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def iterations(self) -> int:
        assert self.losses is not None
        return np.sum(~np.isnan(self.losses))


class EarlyStopper:
    def __init__(
            self,
            patience: int = 5,
            delta_min: float = 1e-4,
        ):
        self.patience = patience
        self.delta_min = delta_min
        self.best = float("inf")
        self.count = 0

    def check(self, loss: np.number) -> bool:
        if self.best - loss > self.delta_min:
            self.best = loss
            self.count = 0
        else:
            self.count += 1

        return self.count >= self.patience
    
    def reset(self) -> None:
        self.best = float('inf')
        self.count = 0
