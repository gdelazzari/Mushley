import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils


X, Y = datasets.agaricus_lepiota()
X, Y = utils.shuffle(X, Y)
(X_train, Y_train), (X_test, Y_test) = utils.split(X, Y, 0.8)

lr = LogisticRegression()

lr.fit(X_train, Y_train)

print(lr.score(X_test, Y_test))
