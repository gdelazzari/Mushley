import numpy as np
from typing import Tuple


def shuffle(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffles a dataset.
    """
    assert len(X) == len(Y)

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], Y[idx]


def split(X: np.ndarray, Y: np.ndarray, p: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits a dataset into a training set and a test set, using the split ratio `p`
    which must be within [0.0, 1.0]. This function doesn't randomly shuffle the
    dataset. If needed, the shuffle must be performed beforehand.
    """
    assert 0.0 <= p <= 1.0
    assert len(X) == len(Y)
    
    s = int(len(X) * p)

    X_train = X[:s]
    Y_train = Y[:s]
    X_test = X[s:]
    Y_test = Y[s:]

    return ((X_train, Y_train), (X_test, Y_test))
