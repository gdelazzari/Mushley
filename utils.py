import numpy as np
from typing import Tuple, List, Optional


def shuffle(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffles a dataset.
    """
    assert len(X) == len(Y)

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], Y[idx]

def feature_shuffle(X: np.ndarray) -> Tuple[np.ndarray, np.array]:
    """
    Randomly shuffles the features of a given dataset.
    """

    idx = np.arange(len(X[0]))
    np.random.shuffle(idx)

    return X[:][idx], idx

def split(X: np.ndarray, Y: np.ndarray, p: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits a dataset into a training set and a test set, using the split ratio `p`
    which must be within [0.0, 1.0]. This function doesn't randomly shuffle the
    dataset. If needed, the shuffle must be performed beforehand.
    Also, this doesn't ensure that the two splits are normalized with respect to
    the labels!
    """
    assert 0.0 <= p <= 1.0
    assert len(X) == len(Y)
    
    s = int(len(X) * p)

    X_train = X[:s]
    Y_train = Y[:s]
    X_test = X[s:]
    Y_test = Y[s:]

    return ((X_train, Y_train), (X_test, Y_test))

def prepare_dataset(X: np.ndarray, Y: np.ndarray, p: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits a dataset into a training set and a test set, using the split ratio `p`
    which must be within [0.0, 1.0]. This function does also randomly shuffle the
    dataset and ensures that the two splits are normalized with respect to the labels.
    """
    assert 0.0 <= p <= 1.0
    assert len(X) == len(Y)

    n = X.shape[0]

    n_1 = int(sum(Y))
    n_0 = len(Y) - n_1

    usable_n = min(n_0, n_1) * 2

    n_train = int(usable_n * p)
    n_test = usable_n - n_train

    # ensure both counts are divisible by 2, eventually by subtracting 1 sample
    n_train = n_train - (n_train % 2)
    n_test = n_test - (n_test % 2)

    assert n_train % 2 == 0
    assert n_test % 2 == 0
    assert n_train + n_test <= usable_n

    available_samples = list(range(len(X)))

    X, Y = shuffle(X, Y)

    def pop_with_label(available_samples: List[int], wanted_label: int) -> Optional[int]:
        candidate = next(filter(lambda x: Y[x[1]] == wanted_label, enumerate(available_samples)), None)

        if candidate is not None:
            i, n = candidate
            available_samples.pop(i)
            return n
        else:
            return None
    
    def build_balanced_dataset(available_samples: List[int], count: int) -> Tuple[np.ndarray, np.ndarray]:
        X_res = []
        Y_res = []

        # build up the training dataset
        while len(X_res) < count:
            i_0 = pop_with_label(available_samples, 0)
            i_1 = pop_with_label(available_samples, 1)

            assert i_0 is not None
            assert i_1 is not None

            assert Y[i_0] == 0
            assert Y[i_1] == 1

            X_res.append(X[i_0])
            Y_res.append(Y[i_0])

            X_res.append(X[i_1])
            Y_res.append(Y[i_1])

        assert len(X_res) == count
        assert len(Y_res) == count
        
        return np.array(X_res), np.array(Y_res)

    X_train, Y_train = build_balanced_dataset(available_samples, n_train)
    X_test,  Y_test  = build_balanced_dataset(available_samples, n_test)

    assert X_train.shape == (n_train, X.shape[1])
    assert Y_train.shape == (n_train, )
    assert X_test.shape == (n_test, X.shape[1])
    assert Y_test.shape == (n_test, )

    assert sum(Y_train) == len(Y_train) - sum(Y_train)
    assert sum(Y_test)  == len(Y_test)  - sum(Y_test)

    return ((X_train, Y_train), (X_test, Y_test))

def one_hot_groups(L: List[int]) -> List[Tuple[int, int]]:
    """
    Returns the consecutive groups for one-hot encoded features, assuming each feature
    is sequentially laid out in the feature array without gaps.

    Example: [2, 4, 3] => [(0, 1), (2, 5), (6, 8)]

    The group indices are inclusive.
    """
    groups = []

    i = 0
    for l in L:
        groups.append((i, i + l - 1))
        i += l

    return groups
