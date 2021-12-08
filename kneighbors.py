import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import datasets
import utils


X, Y = datasets.agaricus_lepiota()
X, Y = utils.shuffle(X, Y)
(X_train, Y_train), (X_test, Y_test) = utils.split(X, Y, 0.8)

knc = KNeighborsClassifier(n_neighbors=5)

knc.fit(X_train, Y_train)

print(knc.score(X_test, Y_test))
