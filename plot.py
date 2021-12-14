import numpy as np

import matplotlib.pyplot as plt

import sys


if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} filename.npy")
    quit()

filename = sys.argv[1]

sh = np.load(filename)

plt.barh(y=np.arange(len(sh)), width=list(sh))
plt.show()

# Check on Shapley values: sum must be equal to the total ensemble score
# Check the distribution also
print(f"\nSum of all Shapley values: {np.sum(sh)}\n")
n_bins = 20
plt.hist(sh, bins=n_bins)
plt.show()
