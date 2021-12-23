SAVE_PGF = True

import numpy as np
import datasets
import utils
import os

import matplotlib as mpl

if SAVE_PGF:
    mpl.use('pgf')

import matplotlib.pyplot as plt


def set_size(width_pt, fraction=1, height_ratio=(5**.5-1)/2, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def average_heavysims(basename):
    single = []
    avg = None
    i = 0
    while os.path.isfile(f"heavy-sims/{basename}.{i}.npy"):
        sh = np.load(f"heavy-sims/{basename}.{i}.npy")
        
        if avg is None:
            avg = np.zeros(sh.shape)
        
        avg += sh
        single.append(sh)

        i += 1
    
    avg /= len(single)

    return avg, single


_, _, fds = datasets.agaricus_lepiota()

sh, shs = average_heavysims('22-10000-0.001-0.5')

sh *= 2.0

if SAVE_PGF:
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

fig = plt.figure(figsize=set_size(350, height_ratio=0.8)) # 252
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Estimated Shapley value for each feature")
ax.set_xticks(np.arange(100) * 0.02)
ax.grid(axis='x', color='0.8', linestyle='--', zorder=-100)
ax.barh(
    y=np.arange(len(sh)),
    width=list(sh),
    fill=False,
    hatch='//////',
    tick_label=[fd.name for fd in fds],
    zorder=100
)

fig.tight_layout()

if SAVE_PGF:
    plt.savefig('features.pgf', format='pgf', bbox_inches='tight')
else:
    plt.show()
