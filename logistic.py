import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm

import os.path


def tmc_shapley(X_train, Y_train, X_test, Y_test, n_samples = None, perf_tolerance: float = 0.01, v_init = 0.5, save_results=True):
    n = np.shape(X_train)[1]

    print(f"Number of features: {n}")

    # create a LogisticRegression classifier with the given hyperparameters
    lr = LogisticRegression(max_iter = 1000)

    # obtain the score using all of the features
    lr.fit(X_train, Y_train)
    vD = lr.score(X_test, Y_test)

    # initialize the vector of shapley values for each feature
    shapley = np.zeros(n)

    # if `n_samples` is set to None, use 2n as the default`
    if n_samples is None:
        n_samples = 2 * n

    try:
        for t in tqdm(range(1, n_samples+1), desc="samples", position=0, smoothing=0.0):
            # obtain a permutation of the features, represented as a vector
            # of indexes into the columns of X_train and X_test
            perm = np.arange(n)
            np.random.shuffle(perm)

            # obtain the permutated versions of X_train and X_test
            perm_X_train = X_train[:, perm]
            perm_X_test = X_test[:, perm]

            v = np.zeros(n + 1)

            # NOTE: is this correct? Shouldn't 50% accuracy be assumed?
            v_prev = v_init # suppose to have zero accuracy with no training features

            for j in tqdm(range(1, n + 1), desc="subsets", position=1, leave=False):
                if abs(vD - v_prev) < perf_tolerance:
                    v = v_prev
                else:
                    lr.fit(perm_X_train[:, :j], Y_train)
                    v = lr.score(perm_X_test[:, :j], Y_test)

                shapley[perm[j - 1]] = (t - 1) / t * shapley[perm[j - 1]] + (v - v_prev) / t
                v_prev = v
    
    except KeyboardInterrupt:
        # Allow to break early with Ctrl+C.
        # Since the result in `shapley` is iteratively computed, it is valid
        # even in such case.
        pass

    if save_results:
        basename = f"{n_samples}-{perf_tolerance}-{v_init}"

        i = 0
        while os.path.isfile(f"{basename}.{i}.npy"):
            i += 1
        
        np.save(f"{basename}.{i}.npy", shapley)

    return shapley


X, Y = datasets.agaricus_lepiota()

(X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)

print(f"Using {len(Y_train)} training samples")
print(f"Using {len(Y_test)} test samples")
print(f"(using {len(Y_train) + len(Y_test)} samples out of {len(Y)} available)")

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
