
from models.KNN import KNearestNeighbors

import numpy as np
import pandas as pd
import seaborn as sns

# retrieve dataset
iris_dataset:pd.DataFrame = sns.load_dataset(name='iris')

y = iris_dataset['species'].values
X = iris_dataset.drop(columns='species').values

# split into train and test data
train_size = 0.8
train_mask = np.random.choice(a=[True, False], size=iris_dataset.shape[0], p=(train_size, 1-train_size))

X_train = X[train_mask]
X_test = X[~train_mask]

y_train = y[train_mask]
y_test = y[~train_mask]

# instanciate model
clf = KNearestNeighbors(k=5)

clf.fit(X=X_train, y=y_train)


y_pred = clf.predict(X=X_test)

accuracy = np.sum(a=(y_pred == y_test)) / len(y_test)
print(accuracy)
