#!/bin/env python3

import numpy as np

from sklearn.linear_model import LogisticRegression

import datasets
import utils
from shapley import tmc_shapley

X, Y, L = datasets.agaricus_lepiota()

(X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)

print(f"Using {len(Y_train)} training samples")
print(f"Using {len(Y_test)} test samples")
print(f"(using {len(Y_train) + len(Y_test)} samples out of {len(Y)} available)")

c = LogisticRegression(max_iter=1000)
v = lambda c, X_train, Y_train, X_test, Y_test: c.fit(X_train, Y_train).score(X_test, Y_test)

vD = v(c, X_train, Y_train, X_test, Y_test)
print(f"Score of whole ensemble of features: {vD}\n")

print("TMC-Shapley:")
sh = tmc_shapley(X_train, Y_train, X_test, Y_test, c, v, groups=utils.one_hot_groups(L), n_samples=10000, perf_tolerance=0.001)
print("Done")
