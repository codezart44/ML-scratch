import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import Estimator
from pyml.optimization.line_search.strong_wolfe import StrongWolfe


class BFGS(Optimizer):  # Broyden Fletcher Goldfarb Shanno
    def __init__(
            self,
            iterations_max: int = 100,
            tolerance: float = 1e-4,
        ):
        self.iterations_max = iterations_max
        self.tolerance = tolerance
        self.line_search = StrongWolfe(c1=1e-4, c2=0.9, iterations_max=20)

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        F = params.size
        If = np.eye(F)

        loss = estimator.loss(x, y, params)
        gk   = estimator.gradient(x, y, params)
        tk   = params
        Bk   = np.eye(F)  # inverse hessian approximation

        for i in range(0, self.iterations_max):
            if self.converged(gk) == True:
                break

            pk = - Bk @ gk  # step direction
            alpha, loss, grad = self.line_search.run(estimator, x, y, loss, gk, tk, pk)
            params = params + alpha * pk

            Bk = self.inverse_hessian_update(If, Bk, tk, params, gk, grad)
            gk = grad
            tk = params

            losses[i] = loss

        self.losses = losses
        return params

    def inverse_hessian_update(
            self, 
            If: NDArray, Bk: NDArray,
            tk: NDArray, tk1: NDArray, 
            gk: NDArray, gk1: NDArray,
        ) -> NDArray:
        sk = tk1 - tk
        yk = gk1 - gk
        ys = yk @ sk
        assert ys > 0, ys
        rhok = 1.0 / ys
        Ak = If - rhok * np.outer(yk, sk)
        Ck = rhok * np.outer(sk, sk)
        Bk = Ak.T @ Bk @ Ak + Ck
        return Bk
        
    def converged(self, gk: NDArray) -> bool:
        return gk @ gk <= self.tolerance ** 2


class LBFGS(Optimizer):  # Low memory Broyden Fletcher Goldfarb Shanno
    def __init__(
            self,
            m: int,
            iterations_max: int = 1000,
            tolerance: float = 1e-4,
        ):
        self.m = m
        self.iterations_max = iterations_max
        self.tolerance = tolerance

        self.line_search = StrongWolfe(c1=1e-4, c2=0.9, iterations_max=20)

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        history = []

        loss = estimator.loss(x, y, params)
        gk   = estimator.gradient(x, y, params)
        tk   = params

        for i in range(0, self.iterations_max):
            if self.converged(gk) == True:
                break

            pk = self.step_direction(gk, history)
            alpha, loss, grad = self.line_search.run(estimator, x, y, loss, gk, tk, pk)
            params = params + alpha * pk

            history.insert(0, self.record(tk, params, gk, grad))
            history = history[:self.m]
            gk = grad
            tk = params

            losses[i] = loss

        self.losses = losses
        return params

    def step_direction(  # two tower algorithm
            self,
            gk: NDArray,
            history: list[tuple[NDArray, NDArray, float]],
        ) -> NDArray:
        
        m = min(self.m, len(history))
        q = gk.copy()
        alpha = np.empty(m, dtype=float)

        for j in range(0, m, 1):
            sj, yj, rhoj = history[j]
            aj = rhoj * sj @ q
            q -= aj * yj
            alpha[j] = aj
        if len(history) >= 1:
            _, y1, rho1 = history[0]
            gammak = 1.0 / (rho1 * (y1 @ y1))
            q *= gammak
        for j in range(m-1, -1, -1):
            sj, yj, rhoj = history[j]
            bj = rhoj * yj @ q
            aj = alpha[j]
            q += (aj - bj) * sj

        return -q

    def record(
            self,
            tk: NDArray, tk1: NDArray,
            gk: NDArray, gk1: NDArray,
        ) -> tuple[NDArray, NDArray, float]:
        sk = tk1 - tk
        yk = gk1 - gk
        ys = yk @ sk
        assert ys > 0, ys
        rhok = 1 / ys
        return (sk, yk, rhok)

    def converged(self, gk: NDArray) -> bool:
        return gk @ gk <= self.tolerance ** 2
