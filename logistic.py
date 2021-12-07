import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm

def tmc_shapley(X_train, Y_train, X_test, Y_test):
    n = np.shape(X_train)[1]

    print(f"Number of features: {n}")

    # create a LogisticRegression classifier with the given hyperparameters
    lr = LogisticRegression(max_iter = 1000)

    # initialize the vector of shapley values for each feature
    shapley = np.zeros(n)

    try:
        for t in tqdm(range(1, 2*n), desc="samples", position=0):
            # obtain a permutation of the features, represented as a vector
            # of indexes into the columns of X_train and X_test
            perm = np.arange(n)
            np.random.shuffle(perm)

            # obtain the permutated versions of X_train and X_test
            perm_X_train = X_train[:, perm]
            perm_X_test = X_test[:, perm]

            v = np.zeros(n + 1)

            # NOTE: is this correct? Shouldn't 50% accuracy be assumed?
            v[0] = 0 # suppose to have zero accuracy with no training features

            for j in tqdm(range(1, n), desc="subsets", position=1, leave=False):
                if False: # implement performance threshold to neglect unimportant features
                    pass
                else:
                    lr.fit(perm_X_train[:, :j], Y_train)
                    v[j] = lr.score(perm_X_test[:, :j], Y_test)

                shapley[perm[j]] = (t - 1) / t * shapley[perm[j]] + (v[j] - v[j - 1]) / t
    
    except KeyboardInterrupt:
        # Allow to break early with Ctrl+C.
        # Since the result in `shapley` is iteratively computed, it is valid
        # even in such case.
        pass

    return shapley


X, Y = datasets.agaricus_lepiota()
X, Y = utils.shuffle(X, Y)

(X_train, Y_train), (X_test, Y_test) = utils.split(X, Y, 0.8)

lr = LogisticRegression()

lr.fit(X_train, Y_train)

print(lr.score(X_test, Y_test))

print("TMC-Shapley:")
sh = tmc_shapley(X_train, Y_train, X_test, Y_test) 
plt.barh(y=np.arange(len(sh)), width=list(sh))
plt.show()
