from pathlib import Path

import numpy as np
import tensorflow as tf
from gph import ripser_parallel as ripser

from ruben import PersistenceDiagram, PersistenceDiagramPoint
import samplers
import utils
from utils.timertree import timer

DATA_SAMPLE_SIZE = 2000
UPPER_DIM = 1
DATA_PATH = Path(
    "/Users/otis/Documents/rubens_speelhoekje/google/google_data/public_data/input_data/task1_v4")
OUTDIR = Path('').parent / 'out'


def compute_distance_matrix(activations):
    with np.errstate(invalid='ignore'):
        correlations = np.nan_to_num(np.corrcoef(activations), copy=False)
    np.fill_diagonal(correlations, 1)
    assert not (correlations < -1).any() or (1 < correlations).any()
    return 1 - np.abs(correlations)


def pd_from_distances(distance_matrix):
    ripser_diagram = ripser(distance_matrix, maxdim=UPPER_DIM,
                            metric="precomputed", n_threads=-1)['dgms']
    diagram = PersistenceDiagram()
    for dim in range(len(ripser_diagram)):
        for point in ripser_diagram[dim]:
            diagram.add_point(PersistenceDiagramPoint(dim, point[0], point[1]))
    return diagram


def pd_from_model(model, x, sampler):

    with timer('activations'):
        activation_samples = sampler(model, x)

    with timer('distances'):
        distance_matrix = compute_distance_matrix(activation_samples)

    with timer('pds'):
        pd = pd_from_distances(distance_matrix)

    return pd


if __name__ == "__main__":
    tf.random.set_seed(1234)
    np.random.seed(1234)
    tf.config.experimental.enable_op_determinism()

    x_train = utils.import_and_sample_data(
        DATA_PATH / "dataset_1", DATA_SAMPLE_SIZE)

    with timer('20'):
        model = utils.import_model(DATA_PATH / "model_20")
        pd = pd_from_model(model, x_train, samplers.random(3000))

    utils.save_data(pd, OUTDIR / 'task1/pds/model20.bin')
    timer.save(OUTDIR / 'task1/timer')
