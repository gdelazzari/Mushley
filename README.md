# Mushley

Applying Shapley values to mushroom edibility classification.

## TODOs
See [issues](https://github.com/gdelazzari/Mushley/issues).

## Current repository structure
The following Python modules are implemented:
- `datasets.py` provides a way to load the [ics.uci.edu mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).
- `utils.py` provides some utility methods (shuffle, split, ...) for working with datasets.
- `logistic.py` tests a LogisticRegression classifier (from scikit-learn) and computes the Shapley value for each feature.
- `kneighbors.py` tests a KNeighborsClassifier (from scikit-learn) and computes the Shapley value for each feature.
- `shapley.py` implements the Truncated Monte Carlo method for approximating Shapley values in a generic way.
- `agaricus-lepiota.data` is the actual dataset which is included in the repository for convenience.

## Getting started
The only meaningful thing to do right now is

```console
$ python logistic.py
$ python kneighbors.py
```

to verify the performance of the classifiers.

## References
[1]: Ghorbani, A. and Zou, J., “Data Shapley: Equitable Valuation of Data for Machine Learning”, *arXiv e-prints*, 2019
