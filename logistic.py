import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm

def tmc_shapley(X, Y):
    n = np.shape(X)[1]
    print("number of features: ", np.shape(X))
    (X_train, Y_train), (X_test, Y_test) = utils.split(X, Y, 0.8)
    lr = LogisticRegression(max_iter = 1000)
    v = np.zeros((n+1,1))
    shapley = np.zeros((n+1,1))

    for t in tqdm(range(1, 2*n), desc="samples", position=0):
        X, perm = utils.feature_shuffle(X)
        perm = perm+1
        v[0] = 0 # suppose to have zero accuracy with no training feature
        for j in tqdm(range(1, n), desc="subsets", position=1, leave=False):
            if False: # implement performance threshold to neglect unimportant features
                pass
            else:
                lr.fit(X_train[:, :j], Y_train)
                v[j] = lr.score(X_test[:, :j], Y_test)

        shapley[perm] = (t - 1) / t * shapley[perm] + (v[1:n+1] - v[0:n]) / t
    return shapley


X, Y = datasets.agaricus_lepiota()
X, Y = utils.shuffle(X, Y)

(X_train, Y_train), (X_test, Y_test) = utils.split(X, Y, 0.8)

lr = LogisticRegression()

lr.fit(X_train, Y_train)

print(lr.score(X_test, Y_test))

print("TMC-Shapley:")
sh = tmc_shapley(X, Y) 
plt.barh(y=np.arange(127), width=list(sh[:, 0]))
plt.show()
