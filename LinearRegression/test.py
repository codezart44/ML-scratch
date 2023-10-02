
import numpy as np
import matplotlib.pyplot as plt

from models.LinRegGD import LinearRegressionGD
from models.LinRegOLS import LinearRegressionOLS




# a = np.array([
#     [1,1,1,1],
#     [1,1,1,1]
# ])
# b = np.array([
#     [2,2],
#     [2,2],
#     [3,3],
#     [2,2]
# ])
# print(a @ b)

# quit()


m_samples = 100

noise = (50 * np.random.randn(m_samples))
X = np.linspace(0, m_samples-1, m_samples).reshape(-1, 1)
w1, w2 = 3, 3
x = X.ravel()
y = 100 + w1 * x + noise
X = (X-np.mean(X))/np.std(X)        ## NOTE really shows the importance of scaling data
y = (y-np.mean(y))/np.std(y)

linregOLS = LinearRegressionOLS()
linregGD = LinearRegressionGD()

linregOLS.fit(X=X, y=y)
linregGD.fit(X=X, y=y)
print(linregOLS.b, linregOLS.w)
print(linregGD.b, linregGD.w)
# quit()


y_predOLS = linregOLS.predict(X=X)
y_predGD = linregGD.predict(X=X)

def mse(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

print(mse(y_true=y, y_pred=y_predOLS))
print(mse(y_true=y, y_pred=y_predGD))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(111)
ax.scatter(x=X.ravel(), y=y, c='r')
ax.plot(X, y_predOLS, c='b', label='OLS')
ax.plot(X, y_predGD, c='g', label='GD')
ax.legend()
plt.show()