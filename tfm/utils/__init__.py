import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


def farthest(d, p):
	d = np.triu(d, 1) + np.triu(d, 1).T
	N = len(d)
	maxdist  = 0
	bestpair = ()
	for i in range(N):
		for j in range(i+1,N):
			if d[i,j]>maxdist:
				maxdist = d[i,j]
				bestpair = (i,j)

	P = set()
	P.add(bestpair[0])
	P.add(bestpair[1])

	while len(P) < p:
		maxdist = 0
		vbest = None
		for v in range(N):
			if v in P:
				continue
			for vprime in P:
				if d[v,vprime]>maxdist:
					maxdist = d[v,vprime]
					vbest   = v
		P.add(vbest)
	return list(P)

class taskdir:
    def __init__(self, task: int):
        self.task = task

    def __rtruediv__(self, path: Path):
        try:
            return next(path.glob(f'task{self.task}*'))
        except StopIteration:
            raise FileNotFoundError

def import_model(path: Path) -> tf.keras.Sequential:
    model = create_model_instance(path / 'config.json')
    if (path / 'weights_init.hdf5').exists():
        model.load_weights(path / 'weights_init.hdf5')
        model.initial_weights = model.get_weights()
    model.load_weights(path / 'weights.hdf5')

    return model


def import_and_sample_data(path: Path, size, entropy=None) -> Tuple[np.ndarray, int]:
    dataset = tf.data.TFRecordDataset([*path.glob('train/shard_*.tfrecord')])
    array = np.array([*dataset.map(_deserialize_example).as_numpy_iterator()])
    # sample = np.random.choice(
    #     [*dataset.map(_deserialize_example).as_numpy_iterator()],
    #     size, replace=False)
    # train = dataset.map(_deserialize_example)
    # ret, _ = map(np.array, zip(
    # 	*train.shuffle(2000, reshuffle_each_iteration=True).take(2000).as_numpy_iterator()))
    # ret = np.array([*train.shuffle(2000, reshuffle_each_iteration=True).take(2000).as_numpy_iterator()])
    seeds = np.random.SeedSequence(entropy)
    sample_ix = np.random.default_rng(seeds).choice(
        len(array), size, replace=False)
    return array[sample_ix].copy(), int(seeds.entropy)


def save(type: str, array: np.ndarray, path: Path):
    assert type in ('pd', 'dm')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(f'.{type}.npy'), array)


def load(type: str, path: Path):
    assert type == 'dm'
    return np.load(path.with_suffix(f'.dm.npy'))


def create_model_instance(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_instance = _model_def_to_keras_sequential(config['model_config'])
    model_instance.build([0] + config['input_shape'])
    return model_instance


def _model_def_to_keras_sequential(model_def):
    """Convert a model json to a Keras Sequential model.

    Args:
        model_def: A list of dictionaries, where each dict describes a layer to add
            to the model.

    Returns:
        A Keras Sequential model with the required architecture.
    """

    def _cast_to_integer_if_possible(dct):
        dct = dict(dct)
        for k, v in dct.items():
            if isinstance(v, float) and v.is_integer():
                dct[k] = int(v)
        return dct

    def parse_layer(layer_def):
        layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
        kwargs = dict(layer_def)
        del kwargs['layer_name']
        return layer_cls(**_cast_to_integer_if_possible(kwargs))

    return tf.keras.Sequential([parse_layer(l) for l in model_def])


def _deserialize_example(serialized_example):
    record = tf.io.parse_single_example(
        serialized_example,
        features={
            'inputs': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string)
        })
    return tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
