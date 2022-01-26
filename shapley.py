import numpy as np

from tqdm import tqdm

import os.path
from typing import Optional, Callable, TypeVar, List, Tuple


C = TypeVar("C")


def tmc_shapley(
    X_train,
    Y_train,
    X_test,
    Y_test,
    classifier: C,
    v_func: Callable[[C, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    groups: Optional[List[Tuple[int, int]]] = None,
    n_samples: Optional[int] = None,
    perf_tolerance: float = 0.01,
    v_init = 0.0,
    save_results=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses the Truncated Monte Carlo algorithm to estimate the Shapley value for
    each feature. Besides the two sample sets, it is required to provide:
    - `classifier`, which is the classifier to be used
    - `v_func`, which gets called as v_func(c, X_train, Y_train, X_test, Y_test) and is
       expected to return the score. The classifier provided as parameter is
       passed as the first parameter.
    
    Additionally, some parameters can be tuned:
    - `groups`: a list of grouped adjacent features can be provided; for instance by
       passing groups=[(0, 2), (3, 4)] the Shapley value of the features {0, 1, 2} is
       jointly computed, same for the features {3, 4}; note that this list must "slice"
       the entire set of features, i.e. include all the features, with no overlaps, in
       ascending order; also, the intervals do include their extremes: an example of a
       valid `groups` partition is [(0, 3), (4, 4), (5, 8), (9, 10)] for a set of 11
       features. By default a grouping in which every feature is by itself is used.
    - `n_samples` is the number of samples to compute, and defaults to 2*n where `n`
      is the number of groups
    - `perf_tolerance` is a threshold for early termination of the samples evaluation
    - `v_init` is the value we assume for V({})
    - `save_results`, if True, dumps the results into a npy file

    A NumPy array with the Shapley value associated to each feature is returned along
    with a NumPy array with the variance for each value.
    """
    m = np.shape(X_train)[1]

    # if `groups` is set to None, build up a grouping where all the features are
    # by themselves
    if groups is None:
        groups = [(i, i) for i in range(m)]
    
    n = len(groups)

    # ensure the groups array is valid
    prev_end = -1
    for (s, e) in groups:
        assert s == prev_end + 1
        assert s <= e
        assert e < m
        prev_end = e
    assert prev_end == m - 1

    # if `n_samples` is set to None, use 2n as the default`
    if n_samples is None:
        n_samples = 2 * n

    print(f"Number of features: {m}")
    print(f"Number of groups:   {n}")

    # obtain the score using all of the features
    vD = v_func(classifier, X_train, Y_train, X_test, Y_test)

    # initialize the vector of shapley values for each group of features
    shapley = np.zeros(n)

    # initialize, for each Shapley value, a list of observed characteristics; this is needed
    # to compute the variance at the end
    shapley_chi = [[] for _ in range(n)]

    try:
        for t in tqdm(range(1, n_samples+1), desc="samples", position=0, smoothing=0.0):
            # obtain a permutation of the groups, represented as a vector
            # of indexes into the `groups` list
            perm_groups = np.arange(n)
            np.random.shuffle(perm_groups)

            perm_groups_indices = np.array(groups, dtype=object)[perm_groups]

            perm_features = np.zeros(m, dtype=int)
            perm_groups_ends = []
            i = 0
            for (a, b) in perm_groups_indices:
                gs = (b + 1) - a
                perm_features[i : i + gs] = np.arange(a, b + 1)
                i += gs
                perm_groups_ends.append(i)
            assert i == m
            assert len(perm_groups_ends) == n

            # obtain the permutated versions of X_train and X_test
            perm_X_train = X_train[:, perm_features]
            perm_X_test = X_test[:, perm_features]

            # use the provided initial value for v
            v_prev = v_init

            for j in tqdm(range(1, n + 1), desc="subsets", position=1, leave=False):
                e = perm_groups_ends[j - 1]

                if abs(vD - v_prev) < perf_tolerance:
                    v = v_prev
                else:
                    v = v_func(classifier, perm_X_train[:, :e], Y_train, perm_X_test[:, :e], Y_test)
                
                chi = v - v_prev

                shapley[perm_groups[j - 1]] = (t - 1) / t * shapley[perm_groups[j - 1]] + chi / t
                shapley_chi[perm_groups[j - 1]].append(chi)

                v_prev = v
        
        # if early termination (by Ctrl+C, see below) didn't happen, we expect for each feature to
        # have collected exactly `n_samples` characteristics
        for i in range(n):
            assert len(shapley_chi[i]) == n_samples
    
    except KeyboardInterrupt:
        # Allow to break early with Ctrl+C.
        # Since the result in `shapley` is iteratively computed, it is valid
        # even in such case.
        pass
    
    # compute the variance of the estimation
    shapley_var = np.zeros(n)
    for i in range(n):
        mu = shapley[i]
        shapley_var[i] = np.var(np.array(shapley_chi[i]) - mu) / len(shapley_chi[i])

    if save_results:
        basename = f"{n}-{n_samples}-{perf_tolerance}-{v_init}"

        i = 0
        while os.path.isfile(f"{basename}.{i}.npy"):
            i += 1
        
        np.save(f"{basename}.{i}.npy", shapley)
        np.save(f"{basename}.{i}.var.npy", shapley_var)

    return shapley, shapley_var
