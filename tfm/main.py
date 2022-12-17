import argparse
import logging
from pathlib import Path

import numpy as np
import ray
from gph import ripser_parallel as ripser

from ruben import PersistenceDiagram, PersistenceDiagramPoint
import samplers
import utils
from utils.timertree import timer, save_timer

DATA_SAMPLE_SIZE = 2000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
# SAMPLER = samplers.StratifiedKMeans(3000, 20000)
SAMPLER = samplers.Random(3000)
TIMEOUT = 20


def compute_distance_matrix(activations):
    with np.errstate(invalid='ignore'):
        correlations = np.nan_to_num(np.corrcoef(activations), copy=False)
    np.fill_diagonal(correlations, 1)
    assert not (correlations < -1).any() or (1 < correlations).any()
    return 1 - np.abs(correlations)

@ray.remote
def pd_from_distances(distance_matrix):
    ripser_diagram = ripser(distance_matrix, maxdim=UPPER_DIM,
                            metric="precomputed", n_threads=-1)['dgms']
    diagram = PersistenceDiagram()
    for dim in range(len(ripser_diagram)):
        for point in ripser_diagram[dim]:
            diagram.add_point(PersistenceDiagramPoint(dim, point[0], point[1]))
    return diagram

def pd_from_model_and_save(model, x, pd_path, timeout):
    included_layers = model.layers[1:]

    with timer('activations'):
        activation_samples = samplers.apply(SAMPLER, model, included_layers, x)
        assert activation_samples.shape == (SAMPLER.n, DATA_SAMPLE_SIZE)

    with timer('distances'):
        distance_matrix = compute_distance_matrix(activation_samples)

    pd = pd_from_distances.remote(distance_matrix)
    with timer('pds'):
        finished = ray.wait([pd], timeout=timeout)[0]

    if finished:
        utils.save_pd(pd, pd_path)
    else:
        ray.cancel(pd, force=True)
        logging.warning(f"Timed out on calculating pd for {model_path.name}")
        utils.save_distances(distance_matrix, pd_path)

if __name__ == "__main__":
    # tf.random.set_seed(1234)
    # np.random.seed(1234)
    # tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path)
    parser.add_argument('-o', '--output', default=OUTDIR, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--timeout', default=TIMEOUT, type=int)
    # parser.add_argument('--threads', default=1, type=int)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)

    logging.info(f"Starting experiment with {SAMPLER.name} timing out after {args.timeout}s")
    logging.info(f"Importing data...")

    x_train = utils.import_and_sample_data(
        args.data_path / "dataset_1", DATA_SAMPLE_SIZE)
    outdir = args.output / 'task1' / SAMPLER.name

    logging.info(f"Finished importing data")

    # sort the models according to their number
    model_paths = sorted(args.data_path.glob('model*'),
                         key=lambda p: (len(p.name), p.name))

    for model_path in model_paths:
        pd_path = outdir / f'{model_path.name}'

        if not args.overwrite:
            try:
                file = next(outdir.glob(pd_path.name + "*"))
                logging.info(f'Found {file}, skipping...')
                continue
            except StopIteration:
                pass

        model = utils.import_model(model_path)
        with timer(model_path.name):
            pd = pd_from_model_and_save(model, x_train, pd_path, args.timeout)
