
import numpy as np


class LinearRegressionOLS:
    def __init__(self) -> None:
        '''Linear regression using ordinary leasts squares (OLS)'''
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''## Training

        XB = y does not always have a solution
        instead:
        X.T @ XB = X.T @ y  (project y down onto the X-plane?)
        solve:
        B = np.linalg.solve(X.T@y, X.T@X)

        alt. linalg.lstsq() for least squares NOTE look up!
        '''

        m_samples, n_features = X.shape

        X = np.hstack((np.ones(m_samples).reshape(-1,1), X))
        n_features += 1

        rank = np.linalg.matrix_rank(A=X)
        if rank < n_features: raise ValueError('Columns of X are not linearly independent.')

        coeffs = np.linalg.inv(X.T @ X) @ (X.T @ y)
        # coeffs = np.linalg.solve(a=X.T @ X, b=X.T @ y)       ## make sure nop linear independence

        self.b = coeffs[0]
        self.w = coeffs[1:]


    def predict(self, X: np.ndarray) -> np.ndarray:
        ''''''
        y_pred = X @ self.w + self.b
        return y_pred






# mse = 1/N * np.sum(np.square(y-y_hat))      # cost function