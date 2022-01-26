import numpy as np

from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class ALFeature:
    name: str
    num_values: int


def agaricus_lepiota() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ALFeature]]:
    """
    Source:
    https://archive.ics.uci.edu/ml/datasets/Mushroom
    
    The function requires the file 'agaricus-lepiota.data' to be in the current
    folder.
    
    Returns the loaded dataset as a tuple of NumPy arrays, where the first contains
    (for each sample) the concatenation of all the features (which are one-hot encoded),
    the second contains (for each sample) the raw features as integer numbers and the
    the third contains (for each sample) the labels (0 <=> edible, 1 <=> poisonous).
    Also returns a list which contains a small description of each feature as a
    `ALFeature` object.
    """
    FEATURES_NAMES = [
        'cap-shape',
        'cap-surface',
        'cap-color',
        'bruises?',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-root',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'stalk-color-above-ring',
        'stalk-color-below-ring',
        'veil-type',
        'veil-color',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat'
    ]

    MISSING_FEATURE_LETTER = '?'

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
        ['b', 'c', 'u', 'e', 'z', 'r'],
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

    X  = [] # one-hot encoded feature vectors
    Xr = [] # raw feature vectors
    Y  = [] # labels

    with open("agaricus-lepiota.data") as f:
        for line in f:
            label_letter, *features_letters = line.removesuffix('\n').split(',')

            assert type(label_letter) == str
            assert type(features_letters) == list

            assert label_letter in LABEL_LETTERS
            y = LABEL_LETTERS.index(label_letter)

            x = np.zeros(Xn, dtype=float)
            xr = np.zeros(22, dtype=int)

            idx = 0 # track the starting index of the current feature in the one-hot encoded vector

            for i, feature_letter in enumerate(features_letters):
                assert i < 22
                assert feature_letter in FEATURES_LETTERS[i] or feature_letter == MISSING_FEATURE_LETTER

                if feature_letter != MISSING_FEATURE_LETTER:
                    feature_value = FEATURES_LETTERS[i].index(feature_letter)

                    # one-hot encode the feature value into the feature vector X only if it is not
                    # missing (if it is missing, the one-hot vector associated to this feature is left
                    # at zero in all of its components)
                    x[idx + feature_value] = 1.0

                    # store the raw feature
                    xr[i] = feature_value
                else:
                    # if the feature is missing, the one-hot encoding is correct (i.e. all components
                    # are left as zero), but the raw encoding must be handled separately: here we set
                    # the value to -1 to represet the fact that the feature is missing with another
                    # category
                    xr[i] = -1

                idx += len(FEATURES_LETTERS[i])
     
            X.append(x)
            Xr.append(xr)
            Y.append(y)
    
    assert len(X) == len(Y)
    assert len(Xr) == len(Y)

    fds = [ALFeature(name, len(fl)) for name, fl in zip(FEATURES_NAMES, FEATURES_LETTERS)]
    
    return np.array(X, dtype=float), np.array(Xr, dtype=int), np.array(Y), fds


if __name__ == "__main__":
    # Test loading the dataset
    X, Y, _ = agaricus_lepiota()
    N = len(X)
    count_1 = len(list(filter(lambda y: y == 1, Y)))

    # Print its length and the percentage of samples with label 1 (poisonous)
    print(N)
    print(count_1 / N)
