import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from gph import ripser_parallel as ripser

from ruben import PersistenceDiagram, PersistenceDiagramPoint
import samplers
import utils
from utils.timertree import timer, save_timer

DATA_SAMPLE_SIZE = 2000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
OVERWRITE = False
SAMPLER = samplers.StratifiedKMeans(3000, 20000)
TIMEOUT = 1000


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


def pd_from_model(model, x):
    included_layers = model.layers[1:]

    with timer('activations'):
        activation_samples = samplers.apply(SAMPLER, model, included_layers, x)
        assert activation_samples.shape == (SAMPLER.n, DATA_SAMPLE_SIZE)

    with timer('distances'):
        distance_matrix = compute_distance_matrix(activation_samples)

    with timer('pds'):
        pd = pd_from_distances(distance_matrix)

    return pd

if __name__ == "__main__":
    # tf.random.set_seed(1234)
    # np.random.seed(1234)
    # tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path)
    parser.add_argument('-o', '--output', default=OUTDIR, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--timeout', default=TIMEOUT, type=int)
    parser.add_argument('--threads', default=1, type=int)
    args = parser.parse_args()

    x_train = utils.import_and_sample_data(
        args.data_path / "dataset_1", DATA_SAMPLE_SIZE)
    outdir = OUTDIR / 'task1' / SAMPLER.name

    model_paths = sorted(args.data_path.glob('model*'), key=lambda p: (len(p.name), p.name))

    try:
        for model_path in model_paths:
            pd_path = outdir / f'{model_path.name}.bin'
            if model_path.name in ["model_156", "model_157", "model_158", "model_220", "model_221"]:
                print(f'Asked to skip {model_path.name}')
                continue

            if not OVERWRITE and pd_path.exists():
                # logging?
                print(f'{pd_path.name} already exists, skipping...')
                continue

            with timer(model_path.name):
                model = utils.import_model(model_path)
                pd = pd_from_model(model, x_train)

            utils.save_data(pd, pd_path)
    finally:
        save_timer(outdir / 'timer')
