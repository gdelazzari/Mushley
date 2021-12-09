import numpy as np

from typing import Tuple, List


def agaricus_lepiota() -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Source:
    https://archive.ics.uci.edu/ml/datasets/Mushroom
    
    The function requires the file 'agaricus-lepiota.data' to be in the current
    folder.
    
    Returns the loaded dataset as a tuple of NumPy arrays, where the first contains
    (for each sample) the concatenation of all the features (which are one-hot encoded)
    and the second contains the labels (0 <=> edible, 1 <=> poisonous). Also returns a
    list which contains the number of components for the one-hot encoding of each feature.
    """
    FEATURES_LETTERS = [
        ['b', 'c', 'x', 'f', 'k', 's'],
        ['f', 'g', 'y', 's'],
        ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
        ['t', 'f'],
        ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
        ['a', 'd', 'f', 'n'],
        ['c', 'w', 'd'],
        ['b', 'n'],
        ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
        ['e', 't'],
        ['b', 'c', 'u', 'e', 'z', 'r', '?'],
        ['f', 'y', 'k', 's'],
        ['f', 'y', 'k', 's'],
        ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        ['p', 'u'],
        ['n', 'o', 'w', 'y'],
        ['n', 'o', 't'],
        ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
        ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
        ['a', 'c', 'n', 's', 'v', 'y'],
        ['g', 'l', 'm', 'p', 'u', 'w', 'd'],
    ]

    LABEL_LETTERS = ['e', 'p']

    assert len(FEATURES_LETTERS) == 22
    assert len(LABEL_LETTERS) == 2

    # compute number of features after one-hot encoding
    Xn = sum([len(c) for c in FEATURES_LETTERS])

    X = []
    Y = []
    with open("agaricus-lepiota.data") as f:
        for line in f:
            label_letter, *features_letters = line.removesuffix('\n').split(',')

            assert type(label_letter) == str
            assert type(features_letters) == list

            assert label_letter in LABEL_LETTERS
            y = LABEL_LETTERS.index(label_letter)

            x = np.zeros(Xn, dtype=float)
            idx = 0 # track the starting index of the current feature
            for i, feature_letter in enumerate(features_letters):
                assert i < 22
                assert feature_letter in FEATURES_LETTERS[i]

                feature_value = FEATURES_LETTERS[i].index(feature_letter)

                # one-hot encode the feature value into the feature vector X
                x[idx + feature_value] = 1.0

                idx += len(FEATURES_LETTERS[i])
     
            X.append(x)
            Y.append(y)
    
    assert len(X) == len(Y)

    L = [len(fl) for fl in FEATURES_LETTERS]
    
    return np.array(X, dtype=float), np.array(Y), L


if __name__ == "__main__":
    # Test loading the dataset
    X, Y, _ = agaricus_lepiota()
    N = len(X)
    count_1 = len(list(filter(lambda y: y == 1, Y)))

    # Print its length and the percentage of samples with label 1 (poisonous)
    print(N)
    print(count_1 / N)
