import numpy as np

from sklearn.tree import DecisionTreeClassifier

import datasets
import utils
from shapley import tmc_shapley
import matplotlib.pyplot as plt


X, Y, fds = datasets.agaricus_lepiota_flat()
L = [fd.num_values for fd in fds]

c = DecisionTreeClassifier()
v = lambda c, X_train, Y_train, X_test, Y_test: 2 * (c.fit(X_train, Y_train).score(X_test, Y_test) - 0.5)

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
sh, sh_var = tmc_shapley(X_train, Y_train, X_test, Y_test, c, v, n_samples=100000, perf_tolerance=0.001, v_init=0.0)
sh_devstd = np.sqrt(sh_var)
print(sh)
print(sh_var)
print(sh_devstd * 2)
plt.barh(y=np.arange(len(sh)), xerr=np.sqrt(sh_var)*2, width=list(sh), capsize=3)
plt.show()

# Check on Shapley values: sum must be equal to the total ensemble score
print(f"\nSum of all Shapley values: {np.sum(sh)}\n")
