"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - UTILS

Plot utils functions.
"""

__all__ = [
    'annotate_heatmap',
    'get_thousands_int_dot_sep',
    'get_transparency_from_data',
    'heatmap'
]

from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_thousands_int_dot_sep(num: int) -> str:
    """
    Get numerical string with dos on thousands.

    :param num: Number
    :return: Str
    """
    return format(num, ',d')


def get_transparency_from_data(data: 'np.ndarray') -> float:
    """
    Returns the best transparency index from data.

    :param data: Data
    :return: Transparency value
    """
    x: int = max(np.shape(data))  # Number of items

    # Define (x1,y1), (x2, y2) curve
    x1: int = 30000
    y1: float = 0.2
    x2: int = 400000
    y2: float = 0.01

    # Limits
    lim = (0.001, 0.5)
    alpha = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return max(lim[0], min(lim[1], alpha))


# noinspection PyDefaultArgument
def heatmap(
        data: 'np.ndarray',
        row_labels: List[str],
        col_labels: List[str],
        ax=None,
        cbar=True,
        cbar_kw={},
        cbarlabel='',
        cbarfontsize=10,
        gridwidth=3,
        gridcolor='w',
        tickrotationha='right',
        hidespine=True,
        **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar
        If False, disables the bar
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
    cbarlabel
        The label for the colorbar. Optional.
    cbarfontsize
        Fontsize of colorbar
    gridwidth
        The width of the grid
    gridcolor
        The color of the grid
    tickrotationha
        Tick rotation, can be 'right' or 'left'
    hidespine
        Hides the spine
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heat map
    im = ax.imshow(data, **kwargs)

    # Create color bar
    if cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom', fontsize=cbarfontsize)
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=-30, ha=tickrotationha,
             rotation_mode='anchor')

    # Turn spines off and create white grid
    if hidespine:
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

    if gridwidth > 0:
        ax.grid(which='minor', color=gridcolor, linestyle='-', linewidth=gridwidth)
    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im, cbar


# noinspection PyDefaultArgument
def annotate_heatmap(
        im,
        data=None,
        valfmt='{x:.2f}',
        textcolors=['black', 'white'],
        threshold=None,
        **textkw
) -> List[str]:
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **textkw
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        # noinspection PyArgumentList
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by text kw
    val_repi = textkw.pop('val_replace', '')
    val_repo = textkw.pop('val_replace_to', '')
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        # noinspection PyUnresolvedReferences
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel"
    # Change the text's color depending on the data
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # noinspection PyCallingNonCallable
            text = im.axes.text(j, i, str(valfmt(data[i, j], None)).replace(val_repi, val_repo), **kw)
            texts.append(text)

    return texts
