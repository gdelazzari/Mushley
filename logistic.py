import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
from shapley import tmc_shapley
import matplotlib.pyplot as plt


X, Y, fds = datasets.agaricus_lepiota()
L = [fd.num_values for fd in fds]

c = LogisticRegression(max_iter=1000)
v = lambda c, X_train, Y_train, X_test, Y_test: c.fit(X_train, Y_train).score(X_test, Y_test)

global vD, X_train, Y_train, X_test, Y_test

(X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)
vD = v(c, X_train, Y_train, X_test, Y_test)

# Ensure we pick a specific split & shuffle of the dataset that guarantees full score
# when using all of the features
while vD < 1.0:
    print(f"This specific shuffle obtained vD = {vD}, shuffling again")
    (X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)
    vD = v(c, X_train, Y_train, X_test, Y_test)

print(f"Using {len(Y_train)} training samples")
print(f"Using {len(Y_test)} test samples")
print(f"(using {len(Y_train) + len(Y_test)} samples out of {len(Y)} available)")
print(f"Score of whole ensemble of features: {vD}\n")

print("TMC-Shapley:")
sh, sh_var = tmc_shapley(X_train, Y_train, X_test, Y_test, c, v, groups=utils.one_hot_groups(L), n_samples=1000, perf_tolerance=0.001)
plt.barh(y=np.arange(len(sh)), xerr=np.sqrt(sh_var)*2, width=list(sh), capsize=3)
plt.show()

# Check on Shapley values: sum must be equal to the total ensemble score
# Check the distribution also
print(f"\nSum of all Shapley values: {np.sum(sh)}\n")
n_bins = 20
plt.hist(sh, bins=n_bins)
plt.show()
