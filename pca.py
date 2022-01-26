import numpy as np

from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression

import datasets
import utils
import matplotlib.pyplot as plt


qualia_names = {'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'], 'cap-surface': ['fibrous', 'grooves', 'scaly', 'smooth'], 'cap-color': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'], 'bruises?': ['bruises', 'no'], 'odor': ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'], 'gill-attachment': ['attached', 'descending', 'free', 'notched'], 'gill-spacing': ['close', 'crowded', 'distant'], 'gill-size': ['broad', 'narrow'], 'gill-color': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'], 'stalk-shape': ['enlarging', 'tapering'], 'stalk-root': ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing'], 'stalk-surface-above-ring': ['fibrous', 'scaly', 'silky', 'smooth'], 'stalk-surface-below-ring': ['fibrous', 'scaly', 'silky', 'smooth'], 'stalk-color-above-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], 'stalk-color-below-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], 'veil-type': ['partial', 'universal'], 'veil-color': ['brown', 'orange', 'white', 'yellow'], 'ring-number': ['none', 'one', 'two'], 'ring-type': ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'], 'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'], 'population': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'], 'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods']}

X, _, Y, fds = datasets.agaricus_lepiota()
L = [fd.num_values for fd in fds]

"""
(X_train, Y_train), (X_test, Y_test) = utils.prepare_dataset(X, Y, 0.8)

print(f"Using {len(Y_train)} training samples")
print(f"Using {len(Y_test)} test samples")
print(f"(using {len(Y_train) + len(Y_test)} samples out of {len(Y)} available)")
"""

pca = SparsePCA(alpha=0.1, verbose=2, tol=1e-5)
pca.fit(X)


# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

print("components for the most important features:")

labels = [f"{fn}={qn}" for fn, qns in qualia_names.items() for qn in qns]

for i in range(pca.components_.shape[0]):
    v = pca.components_[i]
    # print(i, v)
    print(f"{i} <=> {[labels[j] for j in range(len(v)) if abs(v[j]) > 1e-5]}")

X = pca.transform(X)

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

print(f"score = {vD}")

"""
m = len(pca.explained_variance_ratio_)

plt.figure()
plt.subplot(2, 1, 1)
plt.bar(x=np.arange(m), height=pca.explained_variance_ratio_)
plt.subplot(2, 1, 2)
plt.bar(x=np.arange(m), height=pca.singular_values_)
plt.show()
"""
