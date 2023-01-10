import logging

from gph import ripser_parallel
import numpy as np
import ray

import utils
from constants import UPPER_DIM, SAMPLER, DATA_SAMPLE_SIZE, VERSION, INCLUDED_LAYERS
from utils.timertree import timer


def from_model_and_save(model, x, pd_path, timeout):
    activation_sample = sample_neurons(model, x)
    distance_matrix = compute_distance_matrix(activation_sample)
    from_dm_and_save(distance_matrix, pd_path, timeout)


def compute_distance_matrix(activations):
    with timer('distances'):
        with np.errstate(invalid='ignore'):
            correlations = np.nan_to_num(np.corrcoef(activations), copy=False)
        np.fill_diagonal(correlations, 1)
        assert not (correlations < -1).any() or (1 < correlations).any()
        return 1 - np.abs(correlations)


@ray.remote
def from_distances(distance_matrix):
    diagrams = ripser_parallel(distance_matrix, maxdim=UPPER_DIM,
                               metric="precomputed", n_threads=-1)['dgms']
    return np.stack([
        np.r_[point, dim]
        for dim in range(UPPER_DIM+1)
        for point in diagrams[dim]
    ])


def sample_neurons(model, x):
    included_layers = model.layers[INCLUDED_LAYERS[VERSION]]
    with timer('activations'):
        activations = SAMPLER.apply(model, included_layers, x)
        # assert activations.shape == (SAMPLER.n, DATA_SAMPLE_SIZE)
        if activations.shape != (SAMPLER.n, DATA_SAMPLE_SIZE):
            logging.info(f"activations shape: {activations.shape}")
    return activations


def from_dm_and_save(distance_matrix, pd_path, timeout):
    pd_future = from_distances.remote(distance_matrix)
    with timer('pds'):
        finished = ray.wait([pd_future], timeout=timeout)[0]

    if finished:
        pd = ray.get(pd_future)
        utils.save('pd', pd, pd_path)
    else:
        ray.cancel(pd_future, force=True)
        logging.warning(f"Timed out on calculating pd for {pd_path.name}")
        utils.save('dm', distance_matrix, pd_path)
