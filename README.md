# Mushley

Applying Shapley values to mushroom edibility classification.

## TODOs
- [ ] maybe check out this other dataset https://mushroom.mathematik.uni-marburg.de/
- [ ] maybe evaluate other (simpler) classifiers
- [ ] apply Shapley values
- [ ] setup LaTeX report

## Current repository structure
The following Python modules are implemented:
- `datasets.py` provides a way to load the [ics.uci.edu mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).
- `utils.py` provides some utility functions (shuffle, split) for working with datasets.
- `logistic.py` trains a LogisticRegression (from scikit-learn) for the mushroom dataset.
- `agaricus-lepiota.data` is the actual dataset which is included in the repository for convenience.

## Getting started
The only meaningful thing to do right now is

```console
$ python logistic.py
```

to verify the performance of the logistic regression classifier.
