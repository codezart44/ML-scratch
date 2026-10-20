import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyml.base import Estimator, Linear
from pyml.base import Optimizer
from pyml.optimization import GradientDescent, StochasticGradientDescent
from pyml.optimization import NewtonCholesky
from pyml.optimization import BFGS, LBFGS


class SoftmaxRegression(Estimator, Linear):
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
        assert isinstance(optimizer, self.compatible_optimizers)
        self.optimizer = optimizer
        self.lamda = lamda

        self.ohe = OneHotEncoder()
        self.F = None  # features
        self.C = None  # classes

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()

        self.ohe.fit(y)
        y_ohe = self.ohe.encode(y)
        C = self.ohe.classes.size
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        self.F = F
        self.C = C

        params = np.zeros((F+1) * (C-1))
        params = self.optimizer.run(self, x, y_ohe, params)
        params = np.c_[np.zeros(F+1), params.reshape(F+1, C-1)]
        self.params = params

    def loss(self, x, y_ohe, theta):
        N = x.shape[0]
        Wm = np.c_[np.zeros(self.F+1), theta.reshape(self.F+1, self.C-1)]
        mu = Wm.mean(axis=1, keepdims=True)

        logPm = log_softmax(x @ Wm)
        loss = -1/N * np.sum(y_ohe * logPm)
        loss += self.lamda * np.sum( (Wm[1:,:]-mu[1:]) ** 2 )
        return loss

    def gradient(self, x, y_ohe, theta):
        N = x.shape[0]
        Wm = np.c_[np.zeros(self.F+1), theta.reshape(self.F+1, self.C-1)]
        mu = Wm.mean(axis=1, keepdims=True)

        Pm = softmax(x @ Wm)
        grad = 1/N * x.T @ (Pm[:,1:] - y_ohe[:,1:])
        grad[1:,:] += 2 * self.lamda * ( Wm[1:,1:] - mu[1:] )
        return grad.ravel()

    def hessian(self, x, y_ohe, theta):
        F, C = self.F, self.C
        N = x.shape[0]
        Wm = np.c_[np.zeros(self.F+1), theta.reshape(self.F+1, self.C-1)]
        Rm = np.eye(F+1)
        Rm[0, 0] = 0.0

        hess = np.zeros((F+1, C-1, F+1, C-1), dtype=float)
        Pm = softmax(x @ Wm)[:,1:]  # [N, C-1]

        for k in range(C-1):
            for s in range(C-1):
                if k == s:
                    diag = ( Pm[:,k] * (1 - Pm[:,s]) )[:, None]  # [N, 1]
                    XDXm = 1/N * x.T @ (diag * x)
                    hess[:,k,:,k] = XDXm + 2 * self.lamda * Rm * (1 - 1/C)
                if k > s:
                    diag = ( - Pm[:,k] * Pm[:,s] )[:, None]      # [N, 1]
                    XDXm = 1/N * x.T @ (diag * x)
                    hess[:,k,:,s] = XDXm   + 2 * self.lamda * Rm * (-1/C)
                    hess[:,s,:,k] = XDXm.T + 2 * self.lamda * Rm * (-1/C)

        return hess.reshape((F+1)*(C-1), (F+1)*(C-1))

    def predict(self, x):
        return self.ohe.decode(self.linear(x))

    def probability(self, x):
        return softmax(self.linear(x))

    def prb(self, x):
        return self.probability(x)


def softmax(logits: NDArray) -> NDArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def log_softmax(logits: NDArray) -> NDArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return shifted - np.log(exp.sum(axis=1, keepdims=True))


class OneHotEncoder:
    def __init__(self):
        self.classes = None

    def fit(self, labels: NDArray) -> None:
        assert labels.ndim == 1, labels.ndim
        self.classes = np.unique(labels)

    def encode(self, labels: NDArray) -> NDArray:
        assert self.classes is not None
        assert labels.ndim == 1, labels.ndim
        index = np.searchsorted(self.classes, labels)
        N = labels.size
        C = self.classes.size

        valid = np.isin(labels, self.classes)
        assert valid.all() == True, np.unique(labels[~valid])

        onehot = np.zeros((N, C), dtype=float)
        onehot[np.arange(N), index] = 1.0
        return onehot

    def decode(self, onehot: NDArray) -> NDArray:
        assert self.classes is not None
        assert onehot.ndim == 2, onehot.ndim
        assert onehot.shape[1] == self.classes.size
        index = np.argmax(onehot, axis=1)
        labels = self.classes[index]
        return labels


# for k in range(C-1):
#     for s in range(C-1):
#         kron_delta = 1.0 if k==s else 0.0
#         diag = ( Pm[:,k] * (kron_delta - Pm[:,s]) )[:, None]  # [N, 1]
#         hess[:,k,:,s] = 1/N * x.T @ (diag * x)
#         hess[:,k,:,s] += 2 * self.lamda * kron_delta * Rm
