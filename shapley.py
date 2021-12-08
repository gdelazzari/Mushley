import numpy as np

from tqdm import tqdm

import os.path
from typing import Optional, Callable, TypeVar


C = TypeVar("C")


def tmc_shapley(
    X_train,
    Y_train,
    X_test,
    Y_test,
    classifier: C,
    v_func: Callable[[C, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    n_samples: Optional[int] = None,
    perf_tolerance: float = 0.01,
    v_init = 0.5,
    save_results=True
) -> np.ndarray:
    """
    Uses the Truncated Monte Carlo algorithm to estimate the Shapley value for
    each feature. Besides the two sample sets, it is required to provide:
    - `classifier`, which is the classifier to be used
    - `v_func`, which gets called as v_func(c, X_train, Y_train, X_test, Y_test) and is
       expected to return the score. The classifier provided as parameter is
       passed as the first parameter.
    
    Additionally, some parameters can be tuned:
    - `n_samples` is the number of samples to compute, and defaults to 2*n where `n`
      is the number of features
    - `perf_tolerance` is a threshold for early termination of the samples evaluation
    - `v_init` is the value we assume for V({})
    - `save_results`, if True, dumps the results into a npy file

    A NumPy array with the Shapley value associated to each feature is returned.
    """
    n = np.shape(X_train)[1]

    print(f"Number of features: {n}")

    # obtain the score using all of the features
    vD = v_func(classifier, X_train, Y_train, X_test, Y_test)

    # initialize the vector of shapley values for each feature
    shapley = np.zeros(n)

    # if `n_samples` is set to None, use 2n as the default`
    if n_samples is None:
        n_samples = 2 * n

    try:
        for t in tqdm(range(1, n_samples+1), desc="samples", position=0, smoothing=0.0):
            # obtain a permutation of the features, represented as a vector
            # of indexes into the columns of X_train and X_test
            perm = np.arange(n)
            np.random.shuffle(perm)

            # obtain the permutated versions of X_train and X_test
            perm_X_train = X_train[:, perm]
            perm_X_test = X_test[:, perm]

            # use the provided initial value for v
            v_prev = v_init

            for j in tqdm(range(1, n + 1), desc="subsets", position=1, leave=False):
                if abs(vD - v_prev) < perf_tolerance:
                    v = v_prev
                else:
                    v = v_func(classifier, perm_X_train[:, :j], Y_train, perm_X_test[:, :j], Y_test)

                shapley[perm[j - 1]] = (t - 1) / t * shapley[perm[j - 1]] + (v - v_prev) / t
                v_prev = v
    
    except KeyboardInterrupt:
        # Allow to break early with Ctrl+C.
        # Since the result in `shapley` is iteratively computed, it is valid
        # even in such case.
        pass

    if save_results:
        basename = f"{n_samples}-{perf_tolerance}-{v_init}"

        i = 0
        while os.path.isfile(f"{basename}.{i}.npy"):
            i += 1
        
        np.save(f"{basename}.{i}.npy", shapley)

    return shapley
