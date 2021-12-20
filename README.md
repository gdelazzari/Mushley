# Mushley

Applying Shapley values to mushroom edibility classification.

## TODOs
See [issues](https://github.com/gdelazzari/Mushley/issues).

## Current repository structure
The following Python modules are implemented:
- `datasets.py` provides a way to load the [ics.uci.edu mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).
- `utils.py` provides some utility methods (shuffle, split, ...) for working with datasets.
- `logistic.py` tests a LogisticRegression classifier (from scikit-learn) and computes the Shapley value for each feature.
- `shapley.py` implements the Truncated Monte Carlo method for approximating Shapley values in a generic way.
- `agaricus-lepiota.data` is the actual dataset which is included in the repository for convenience.
- `plot.py` can be used to plot the `.npy` files produced by the simulations

## Getting started
The only meaningful thing to do right now is

```console
$ python logistic.py
$ python kneighbors.py
```

to verify the performance of the classifiers.

## Simulations on DEI's Blade cluster
To run heavy simulations on our computing cluster, some `.job` files are provided.

After installing the required dependencies with an interactive job as such:
```console
[user@login~]$ interactive
user@runner:~$ pip3 install --user sklearn tqdm
user@runner:~$ exit
```

The steps to launch, monitor and stop a job are something like this:

```console
[user@login~]$ sbatch sim01.job
[user@login~]$ squeue -u $(whoami)                  # check that the job has started
[user@login~]$ tail -n +1 -f sim01_out_<jobid>.txt  # monitor the output
[user@login~]$ scancel --signal=sigint <jobid>      # cancel the job with SIGINT
```

Canceling the job by sending a SIGINT (instead of the default SIGTERM) allows for the `shapley.tmc_shapley`
function to gracefully stop the computation and dump a `.npy` file with the partial estimation using the 
samples computed up until now.

## References
[1]: Ghorbani, A. and Zou, J., “Data Shapley: Equitable Valuation of Data for Machine Learning”, *arXiv e-prints*, 2019

