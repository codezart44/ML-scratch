
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


linregOLS = LinearRegressionOLS()
linregGD = LinearRegressionGD()

linregOLS.fit(X=X, y=y)
linregGD.fit(X=X, y=y)
print(linregOLS.b, linregOLS.w)
print(linregGD.b, linregGD.w)
quit()


y_pred = linregOLS.predict(X=X)

def mse(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

print(mse(y_true=y, y_pred=y_pred))
print(linregOLS.w, linregOLS.b)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(111)
ax.scatter(x=X.ravel(), y=y, c='r')
ax.plot(X, y_pred)
plt.show()