import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator

class NewtonCholesky(Optimizer):
    def __init__(
            self,
            iterations_max: int,
            tolerance: float,
            learning_rate: float | None,
            ):
        self.iterations_max = iterations_max
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)

        for i in range(self.iterations_max):
            Hk = estimator.hessian(x, y, params)
            gk = estimator.gradient(x, y, params)

            if self.convergence(gk) == True:
                break

            Hk = 0.5 * (Hk.T + Hk)  # ensure symmetry
            Lk = np.linalg.cholesky(Hk, upper=False)
            zk = np.linalg.solve(Lk, -gk)
            pk = np.linalg.solve(Lk.T, zk)

            params += self.learning_rate * pk
            loss = estimator.loss(x, y, params)
            losses[i] = loss

        self.losses = losses
        return params

    def convergence(self, gk: NDArray) -> bool:
        return gk.T @ gk <= self.tolerance ** 2
