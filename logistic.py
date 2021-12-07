import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm


def tmc_shapley(X_train, Y_train, X_test, Y_test, perf_tolerance: float = 0.01):
    n = np.shape(X_train)[1]

    print(f"Number of features: {n}")

    # create a LogisticRegression classifier with the given hyperparameters
    lr = LogisticRegression(max_iter = 1000)

    # obtain the score using all of the features
    lr.fit(X_train, Y_train)
    vD = lr.score(X_test, Y_test)

    # initialize the vector of shapley values for each feature
    shapley = np.zeros(n)

    try:
        for t in tqdm(range(1, 2*n), desc="samples", position=0, smoothing=0.0):
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

            for j in tqdm(range(1, n + 1), desc="subsets", position=1, leave=False):
                if abs(vD - v[j - 1]) < perf_tolerance:
                    v[j] = v[j - 1]
                else:
                    lr.fit(perm_X_train[:, :j], Y_train)
                    v[j] = lr.score(perm_X_test[:, :j], Y_test)

                shapley[perm[j - 1]] = (t - 1) / t * shapley[perm[j - 1]] + (v[j] - v[j - 1]) / t
    
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

print(f"Score of whole ensemble of features: {lr.score(X_test, Y_test)}\n")

print("TMC-Shapley:")
sh = tmc_shapley(X_train, Y_train, X_test, Y_test) 
plt.barh(y=np.arange(len(sh)), width=list(sh))
plt.show()

# Check on Shapley values: sum must be equal to the total ensemble score
# Check the distribution also
print(f"\nSum of all Shapley values: {np.sum(sh)}\n")
n_bins = 20
plt.hist(sh, bins=n_bins)
plt.show()