
import numpy as np



class LinearRegressionGD:
    def __init__(self) -> None:
        '''Linear regression using gradient descent (GD)'''
        self.w = None
        self.b = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ''' ## Training
        Find weight (w) and bias (b) that minimizes MSE (cost) of residuals.

        Use Gradient Descent, Jocobian with derivatives of f with respect to w & b:

        >>> df/dw
        >>> df/db

        Init w & b to zero.
        Given a train data:
        - calculate residuals (MSE rather).
        - find new w & b that minimizes MSE using gradient descent.
        - update w & b, rinse and repeat n times. 

        >>> y_hat = wx + b
        '''
        alpha = 0.01       # learning rate
        n = 1000

        m_samples, n_features = X.shape
        X = np.hstack((np.ones(m_samples).reshape(-1, 1), X))
        
        n_features += 1
        coeffs = np.zeros(n_features)

        for _ in range(n):
            y_hat = X @ coeffs
            coeffs -= alpha * 1/m_samples * X.T @ (y-y_hat)

            # dw = (2/N) * X.T @ (y_hat-y)                  # d(wx + b)^2/dw = 2x * (wx + b)
            # db = (2/N) * np.sum(y_hat-y)                  # d(wx + b)^2/db = 2 * (wx + b)

            # w = w - alpha * dw          # add negative gradient to minimize loss as fast as possible
            # b = b - alpha * db

        self.b = coeffs[0]
        self.w = coeffs[1:]


    def predict(self, X: np.ndarray) -> np.ndarray:
        ''''''
        y_pred = X @ self.w + self.b
        return y_pred
    

