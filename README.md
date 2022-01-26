# Mushley

Shapley value for feature evaluation in a mushroom edibility classification problem.

Francesco Lorenzi, Giacomo De Lazzari - Department of Information Engineering, University of Padova

## Report
The code and the work stored here is strictly associated to our project for the Game Theory course held by professor Leonardo Badia at our university.

The report is [included in the repository](./report.pdf).

## Repository structure
The following Python modules are implemented:
- `datasets.py` provides a way to load the [ics.uci.edu mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).
- `utils.py` provides some utility methods (shuffle, split, ...) for working with datasets.
- `logistic.py` tests a LogisticRegression classifier (from scikit-learn) on the one-hot encoded features and computes the Shapley value for each one of them.
- `decisiontree.py` tests a DecisionTreeClassifier (from scikit-learn) and computes the Shapley value for each feature.
- `shapley.py` implements the Truncated Monte Carlo (TMC) method for approximating Shapley values in a generic way.
- `agaricus-lepiota.data` is the actual dataset which is included in the repository for convenience.
- `plot.py` can be used to plot the `.npy` files produced by the simulations
- `report_plot_features.py` can be used to plot the Shapley values at feature granularity.
- `report_plot_qualia.py` can be used to plot the Shapley values at *qualia* (single one-hot encoded vector component) granularity.
- `pca.py` has been used to explore the feature space of the dataset.
- `validate.py` is a quite messy script which has been used to test various assumptions and properties of the obtained experimental results.

## TMC Shapley implementation details
The implementation of the Truncated Monte Carlo algorithm is found in the module `shapley.py`, and has been implemented in a generic way.

Evaluating the Shapley value for groups of consecutive features is also possible by specifying a partition of the feature space as a parameter. This was used, when exploring a linear classifier on one-hot encoded features, to obtain the Shapley values of groups of *qualia*, thus obtaining the values for the single entire feature and not of its sub-components.

The implementation is in pure Python with the help on NumPy where possible.

Given that simulations can run for a long time, the `tmc_shapley` function is interactive in the sense that the progress is shown in real-time with tqdm and `CTRL+C` can be used to safely stop the simulation early, resulting in the samples collected up until now to be used for a partial estimate.

## Getting started
To reproduce the plot obtained in the report, the following steps can be taken

```console
$ python decisiontree.py
$ mkdir heavy-sims
$ mv *.npy heavy-sims/
$ python report_plot_features.py
```

### Workflow in brief
In short, the following scripts

- `logistic.py`
- `decisiontree.py`

run simulations (with different models) and produce `.npy` files with the resulting averages of all the Monte Carlo samples. The filename encodes the simulation parameters (number of samples, early termination condition threshold, ...). "Heavy" simulations results are stored in the special folder `heavy-sims`, from which the `report_plot_*.py` scripts load the raw data from.

## References used in the commit messages
[1]: Ghorbani, A. and Zou, J., “Data Shapley: Equitable Valuation of Data for Machine Learning”, *arXiv e-prints*, 2019
