import numpy as np

from sklearn.neighbors import KNeighborsClassifier

import datasets
import utils
from shapley import tmc_shapley
import matplotlib.pyplot as plt


X, Y, _ = datasets.agaricus_lepiota()

(X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)

print(f"Using {len(Y_train)} training samples")
print(f"Using {len(Y_test)} test samples")
print(f"(using {len(Y_train) + len(Y_test)} samples out of {len(Y)} available)")

c = KNeighborsClassifier(n_neighbors=5)
v = lambda c, X_train, Y_train, X_test, Y_test: c.fit(X_train, Y_train).score(X_test, Y_test)

vD = v(c, X_train, Y_train, X_test, Y_test)
print(f"Score of whole ensemble of features: {vD}\n")

print("TMC-Shapley:")
sh = tmc_shapley(X_train, Y_train, X_test, Y_test, c, v)
plt.barh(y=np.arange(len(sh)), width=list(sh))
plt.show()

# Check on Shapley values: sum must be equal to the total ensemble score
# Check the distribution also
print(f"\nSum of all Shapley values: {np.sum(sh)}\n")
n_bins = 20
plt.hist(sh, bins=n_bins)
plt.show()
