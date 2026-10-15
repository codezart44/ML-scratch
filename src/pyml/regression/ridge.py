import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.base import Estimator, Linear
from pyml.base import Optimizer
from pyml.optimization import GradientDescent, StochasticGradientDescent
from pyml.optimization import NewtonCholesky
from pyml.optimization import BFGS, LBFGS


class Ridge(Estimator, Linear):
    compatible_optimizers = (
        GradientDescent, StochasticGradientDescent,
        NewtonCholesky,
        BFGS, LBFGS,
    )
    def __init__(
            self,
            optimizer: Optimizer,
            lamda: float = 1e-2,
        ):
        self.optimizer = optimizer
        self.lamda = lamda
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        params = np.zeros(F+1)
        params = self.optimizer.run(self, x, y, params)
        self.params = params

    def gradient(self, x: NDArray, y: NDArray, theta: NDArray) -> NDArray:
        N, F = x.shape
        grad = -2/N * x.T @ (y - x @ theta)
        grad[1:] += 2 * self.lamda * theta[1:]
        return grad

    def loss(self, x: NDArray, y: NDArray, theta: NDArray) -> float:
        loss = np.mean((y - x @ theta) ** 2)
        loss += self.lamda * np.sum(theta[1:] ** 2)
        return loss

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)


class RidgeQR(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            ):
        self.lamda = lamda
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        eye = np.eye(F+1)
        eye[0,0] = 0.0
        x_stacked = np.r_[x, np.sqrt(N * self.lamda) * eye]
        y_stacked = np.r_[y, np.zeros(F+1)]
        q, r = np.linalg.qr(x_stacked)
        params = np.linalg.solve(r, q.T @ y_stacked)

        self.params = params

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)


class RidgeCholesky(Estimator, Linear):
    def __init__(
            self,
            lamda: float,
            ):
        self.lamda = lamda
        self.params: NDArray = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        eye = np.eye(F+1)
        eye[0,0] = 0.0
        a = x.T @ x + N * self.lamda * eye
        b = x.T @ y
        l = np.linalg.cholesky(a)
        z = np.linalg.solve(l, b)
        params = np.linalg.solve(l.T, z)

        self.params = params

    def predict(self, x: ArrayLike) -> NDArray:
        return self.linear(x)
