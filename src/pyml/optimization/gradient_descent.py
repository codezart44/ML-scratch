import numpy as np
from numpy.typing import NDArray

from pyml.base import Optimizer
from pyml.base import EarlyStopper


class GradientDescent(Optimizer):
    def __init__(
            self,
            iterations_max: int = 100_000,
            learning_rate: float = 1e-3,
            tolerance: float = 1e-2,
        ):
        self.iterations_max = iterations_max
        self.learning_rate = learning_rate
        self.tolerance = tolerance
    
    def run(self, estimator, x,  y,  params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        
        for i in range(self.iterations_max):
            grad = estimator.gradient(x, y, params)
            params -= self.learning_rate * grad

            loss = estimator.loss(x, y, params)
            losses[i] = loss

            if self.converged(grad) == True:
                break

        self.losses = losses
        return params

    def converged(self, grad: NDArray) -> bool:
        return grad @ grad <= self.tolerance ** 2


# TODO add to GradientDescent
# split_trn_val = 0.2,
class StochasticGradientDescent(Optimizer):
    def __init__(
            self,
            batch_size: int = 32,
            iterations_max: int = 100_000,
            learning_rate: float = 1e-2,
            gamma: float = 1e-2,  # learning rate decay speed
            tolerance: float = 1e-4,
            patience: int = 100,
            seed: int = None,
        ):
        assert batch_size >= 1
        self.batch_size = batch_size
        self.iterations_max = iterations_max
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.early_stopper = EarlyStopper(patience, tolerance)
        self.rng = np.random.default_rng(seed)

    def run(self, estimator, x, y, params):
        losses = np.full((self.iterations_max,), fill_value=np.nan)
        N = x.shape[0]
        assert self.batch_size <= N, N
        index = np.arange(N)
        self.early_stopper.reset()

        for i in range(self.iterations_max):
            self.rng.shuffle(index)
            eta_i = self.learning_rate / (1.0 + self.gamma * i)

            for j in range(0, N, self.batch_size):
                ib = index[j:j+self.batch_size]
                xb = x[ib]
                yb = y[ib]
                grad = estimator.gradient(xb, yb, params)
                params -= eta_i * grad

            loss = estimator.loss(x, y, params)
            losses[i] = loss

            if self.converged(loss) == True:
                break

        self.losses = losses
        return params

    def converged(self, loss: float) -> bool:
        return self.early_stopper.check(loss)
