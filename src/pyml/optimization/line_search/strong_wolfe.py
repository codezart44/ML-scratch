import numpy as np
from numpy.typing import NDArray

from pyml.base import Estimator


class StrongWolfe:
    def __init__(
            self,
            c1: float = 1e-4,
            c2: float = 0.9,
            iterations_max = 20,
        ):
        assert 0 < c1 < c2 < 1, (c1, c2)
        self.c1 = c1
        self.c2 = c2
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
        ) -> tuple[float, float, NDArray]:
        grad_fn = lambda alpha: estimator.gradient(x, y, tk + alpha * pk)
        loss_fn = lambda alpha: estimator.loss(x, y, tk + alpha * pk)
        
        f0 = loss     # f(0), objective function
        d0 = gk @ pk  # d(0), directional derivative

        alpha_lo, alpha_hi, loss, grad = self.bracket(loss_fn, grad_fn, pk, f0, d0)
        if alpha_hi == alpha_lo:
            return alpha_hi, loss, grad
        
        alpha, loss, grad = self.zoom(loss_fn, grad_fn, pk, f0, d0, alpha_lo, alpha_hi)
        return alpha, loss, grad
    
    def bracket(self, loss_fn, grad_fn, pk, f0, d0):
        alpha_lo = 0
        alpha_hi = np.inf
        alpha = 1
        
        for _ in range(self.iterations_max):  # bracket
            loss = loss_fn(alpha)
            grad = grad_fn(alpha)
            fa = loss
            da = grad @ pk

            if not self.armijo(f0, d0, fa, alpha):
                alpha_hi = alpha
                break

            if not self.strong_wolfe(d0, da):
                if da > 0:
                    alpha_hi = alpha
                    break
                else:
                    alpha_lo = alpha
                    alpha *= 2
                    continue

            return alpha, alpha, loss, grad
    
        return alpha_lo, alpha_hi, loss, grad

    def zoom(self, loss_fn, grad_fn, pk, f0, d0, alpha_lo, alpha_hi):
        assert np.isfinite(alpha_hi), alpha_hi

        for _ in range(self.iterations_max):  # zoom
            alpha = 0.5 * (alpha_lo + alpha_hi)  # bisection
            loss = loss_fn(alpha)
            grad = grad_fn(alpha)
            fa = loss
            da = grad @ pk

            if not self.armijo(f0, d0, fa, alpha):
                alpha_hi = alpha
                continue

            if not self.strong_wolfe(d0, da):
                if da > 0:
                    alpha_hi = alpha
                else:
                    alpha_lo = alpha
                continue
            
            return alpha, loss, grad

        raise RuntimeError

    def armijo(self, f0, d0, fa, alpha) -> bool:  
        return fa - f0 <= self.c1 * alpha * d0
        # f(alpha) - f(0) <= c1 alpha f'(0)

    def strong_wolfe(self, d0, da) -> bool:  
        return np.abs(da) <= self.c2 * np.abs(d0)
        # |f'(alpha)| <= c2|f'(0)|
