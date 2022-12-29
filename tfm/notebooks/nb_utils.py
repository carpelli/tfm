from itertools import accumulate
from pathlib import Path
import json
import sys
from types import ModuleType, FunctionType
from gc import get_referents
from typing import Callable, Iterable, TypeVar, Union

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


def annotated_heatmap(data, *, title, x_labels, y_labels):
    # adapted from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, str(round(data[i, j], 3)),
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


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
