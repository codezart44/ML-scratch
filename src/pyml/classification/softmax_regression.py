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

        self.classes = None
        self.F = None  # features
        self.C = None  # classes

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x).copy()
        y = np.asarray(y).copy()
        y_ohe, classes = one_hot_encoding(y)
        C = classes.size
        N, F = x.shape
        x = np.c_[np.ones(N), x]

        self.F = F
        self.C = C

        params = np.zeros((F+1) * C)
        params = self.optimizer.run(self, x, y_ohe, params)
        
        self.params  = params.reshape(F+1, C)
        self.classes = classes

    def loss(self, x, y_ohe, theta):  # TODO add regularization
        N = x.shape[0]
        Wm = theta.reshape(self.F+1, self.C)

        logp = log_softmax(x @ Wm)
        loss = -1/N * np.sum(y_ohe * logp)
        return loss

    def gradient(self, x, y_ohe, theta):
        N = x.shape[0]
        Wm = theta.reshape(self.F+1, self.C)
        
        p = softmax(x @ Wm)
        grad = 1/N * x.T @ (p - y_ohe)
        return grad.ravel()

    def hessian(self, x, y_ohe, theta):
        raise NotImplementedError

    def predict(self, x):
        index = np.argmax(self.linear(x), axis=1)
        return self.classes[index]

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


def one_hot_encoding(y: NDArray) -> tuple[NDArray, NDArray]:
    classes, inverse = np.unique(y, return_inverse=True)
    N = y.size
    C = classes.size
    y_ohe = np.zeros((N, C), dtype=float)
    y_ohe[np.arange(N), inverse] = 1
    return y_ohe, classes
