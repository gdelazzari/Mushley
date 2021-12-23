SAVE_PGF = True

import numpy as np
import datasets
import utils
import os

import matplotlib as mpl

if SAVE_PGF:
    mpl.use('pgf')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


qualia_names = {'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'], 'cap-surface': ['fibrous', 'grooves', 'scaly', 'smooth'], 'cap-color': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'], 'bruises?': ['bruises', 'no'], 'odor': ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'], 'gill-attachment': ['attached', 'descending', 'free', 'notched'], 'gill-spacing': ['close', 'crowded', 'distant'], 'gill-size': ['broad', 'narrow'], 'gill-color': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'], 'stalk-shape': ['enlarging', 'tapering'], 'stalk-root': ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted'], 'stalk-surface-above-ring': ['fibrous', 'scaly', 'silky', 'smooth'], 'stalk-surface-below-ring': ['fibrous', 'scaly', 'silky', 'smooth'], 'stalk-color-above-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], 'stalk-color-below-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'], 'veil-type': ['partial', 'universal'], 'veil-color': ['brown', 'orange', 'white', 'yellow'], 'ring-number': ['none', 'one', 'two'], 'ring-type': ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'], 'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'], 'population': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'], 'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods']}


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


_, _, _, fds = datasets.agaricus_lepiota()

sh, sh_var, shs = utils.average_heavysims('125-10000-0.001-0.5')

sh *= 2.0

if SAVE_PGF:
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

labels = [f"{fn}={qn}" for fn, qns in qualia_names.items() for qn in qns]

L = utils.one_hot_groups([fd.num_values for fd in fds])

fig = plt.figure(figsize=set_size(350, height_ratio=1.4)) # 252
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Estimated Shapley value for each quale")
ax.set_xticks(np.arange(100) * 0.01)
ax.grid(axis='x', color='0.8', linestyle='--', zorder=-100)
ax.barh(
    y=np.arange(len(sh)),
    width=list(sh),
    #xerr=np.sqrt(sh_var)*2,
    #ecolor='red',
    fill=True,
    #hatch='//////',
    #tick_label=labels,
    color='k',
    zorder=100
)
ax.set_yticks([-0.5] + [y + 0.5 for _, y in L])
ax.yaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.FixedLocator([(a + b) / 2 for a, b in L]))
ax.yaxis.set_minor_formatter(ticker.FixedFormatter([f.name for f in fds]))
ax.tick_params('y', length=0, width=0, which='minor')

fig.tight_layout()

#fig.subplots_adjust(left=0.2)

if SAVE_PGF:
    plt.savefig('qualia.pgf', format='pgf', bbox_inches='tight')
else:
    plt.show()
