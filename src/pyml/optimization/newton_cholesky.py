import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator

class NewtonCholesky(Optimizer):
    def __init__(
            self,
            iterations_max: int,
            tolerance: float,
            ):
        self.iterations_max = iterations_max
        self.tolerance = tolerance

    def run(self, estimator, x, y, params):
        # Hk @ pk = -gk
        # LL^T @ pk = -gk
        # Lz = -gk
        # L^T @ pk = z
        # --> pk
        ...

    def convergence(self, gk: NDArray) -> bool:
        ...
