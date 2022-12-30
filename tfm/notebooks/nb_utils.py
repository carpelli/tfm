from itertools import accumulate
from pathlib import Path
import json
import sys
from types import ModuleType, FunctionType
from gc import get_referents
from typing import Callable, Iterable, TypeVar, Union

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

T = TypeVar('T')

outdir = Path('../../out')
data_root = Path('../../data')
task_string = {1: 'task1_v4', 2: 'task2_v1'}
data_path = {
    task: data_root / f'public_data/input_data/{string}'
    for task, string in task_string.items()
}
ref_file = {
    task: data_root / f'public_data/reference_data/{string}/model_configs.json'
    for task, string in task_string.items()
}

gen_gaps = {}
for task in task_string:
    gen_gaps[task] = {
        'model_' + k: v['metrics']['train_acc'] - v['metrics']['test_acc']
        for k, v in json.loads(ref_file[task].read_text()).items()
    }


def sift(iter: Iterable[T], by: Callable[[T], bool]) -> tuple[list[T], list[T]]:
    """Split an iterable into two lists based on a condition

    Args:
        iter (Iterable[T]): Iterable to split
        by (Callable[[T], bool]): Function to test condition

    Returns:
        tuple[list[T], list[T]]: Two lists for true and false conditions respectively
    """
    return ([x for x in iter if by(x)], [x for x in iter if not by(x)])


def get_files(task: int, sampler: Union[str, None] = None, pattern = '*.pd.npy'):
    """Get the pd files either for all experiments with a certain sampler or
    one experiment with all samplers

    Args:
        task (int): Task number
        sampler (Union[str, None], optional): Sampler. Defaults to None.

    Returns:
        Generator of (tuples of sampler name and) Path generators
    """
    taskdir = outdir / ('task' + str(task))
    if sampler:
        return ([*dir.glob(pattern)] for dir in dirs(taskdir / sampler))
    else:
        return ((dir.name, [*dirs(dir)[0].glob(pattern)]) for dir in dirs(taskdir))


def dirs(path: Path):
    return sorted(dir for dir in path.iterdir() if dir.is_dir())


def heatmap(data, row_labels, col_labels, ax=None,
            cbar=False, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    (from matplotlib docs)

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    if col_labels:
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=bool(col_labels), bottom=False,
                   labeltop=bool(col_labels), labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.495, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar if cbar else None


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    (from matplotlib docs)

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
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def getsize(obj):
    """sum size of object & members."""
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError(
            'getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
