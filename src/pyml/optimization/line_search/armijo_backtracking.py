from numpy.typing import NDArray

from pyml.base import Estimator


class ArmijoBacktracking:
    def __init__(
            self,
            c1: float = 1e-4,
            rho: float = 0.5,
            iterations_max = 10,
        ):
        assert 0 < c1 < 1, c1
        assert 0 < rho < 1, rho
        self.c1 = c1
        self.rho = rho
        self.iterations_max = iterations_max

    def run(
            self,
            estimator: Estimator,
            x: NDArray,
            y: NDArray,
            loss: float,
            gk: NDArray,  # gradient
            tk: NDArray,  # params
            pk: NDArray,  # direction
        ) -> tuple[float, float]:
        loss_fn = lambda alpha: estimator.loss(x, y, tk + alpha * pk)

        f0 = loss     # objective function
        d0 = gk @ pk  # directional derivative
        assert d0 < 0, d0  # assumes descending search direction

        alpha, loss = self.backtracking(loss_fn, f0, d0)
        return alpha, loss

    def backtracking(self, loss_fn, f0, d0):
        alpha = 1.0

        for _ in range(self.iterations_max):
            loss = loss_fn(alpha)
            fa = loss

            if not self.armijo(alpha, f0, d0, fa):
                alpha *= self.rho
                continue

            return alpha, loss

        raise RuntimeError

    def armijo(self, alpha, f0, d0, fa) -> bool:
        return fa - f0 < self.c1 * alpha * d0
