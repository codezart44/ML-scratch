
import numpy as np
import matplotlib.pyplot as plt

from models.LogReg import LogisticRegression



X = np.linspace(-49, 50, 100).reshape(-1, 1)
y = np.hstack((np.zeros(75), np.ones(25)))



logreg = LogisticRegression()


logreg.fit(X=X, y=y)

y_pred = logreg.predict(X=X)

print(y_pred)


fig = plt.figure(figsize=(10, 8))
ax: plt.Axes = plt.axes(111)
ax.scatter(x=X, y=y, c='k')
ax.plot(X, y_pred, ls=':', c='r')
plt.show()

