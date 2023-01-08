import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from . import gen_gaps, get_files, cache_path

def features_for_dim(pd, dim):
	# b: birth, d: death
	b, d, _ = pd[pd[:,2] == dim].T
	if dim == 0:
		assert d[-1] == np.inf
		d[-1] = 1
	with np.errstate(invalid='ignore', divide='ignore'):
		return [
			mean_bd := np.c_[b, d].mean(axis=0),
			mean_bd**2,
			np.nan_to_num(1/mean_bd + np.log(mean_bd)), # fix divide by zero
			np.c_[b, d].std(axis=0),
			[np.mean(d - b)],
			[np.mean(d - b)**2],
			[np.mean((b + d) / 2)],
			[np.mean((b + d) / 2)**2],
		]

features = {
	'avg_birth_death': 2,
	'avg_birth_death_squared': 2,
	'avg_birth_death_inverted' : 2,
	'std_birth_death': 2,
	'avg_life': 1,
	'avg_life_squared': 1,
	'avg_half_life': 1,
	'avg_half_life_squared': 1
}

flat_features = [*itertools.chain(*(
	[f] if len == 1 else [f.replace('_death', ''), f.replace('_birth', '')]
	for f, len in features.items()
))]
flat_features = [f + '.0' for f in flat_features] + [f + '.1' for f in flat_features]

combinations = [*itertools.chain(*(
	itertools.combinations(features.keys(), r)
	for r in range(1, len(features) + 1)
))]


def features_all_dims(pd):
	arrs = [np.concatenate(features_for_dim(pd, dim)) for dim in (0, 1)]
	return np.vstack(arrs)


def get_labeled_data(task, files, fn=features_all_dims):
	gaps = gen_gaps(task)
	pds = [*map(np.load, files)]
	X = np.stack([*map(fn, pds)])
	y = np.array([gaps[f.name.split('.')[0]] for f in files])
	return X, y


def feature_mask(included):
	mask = np.tile(np.concatenate([
		np.full(length, 1 if feature in included else 0)
		for feature, length in features.items()
	]), 2)
	return np.where(mask)[0]


def run_experiment(X, y, masks, tries, model_fn):
	X = X.reshape((len(X), -1))
	results = np.empty((len(masks), tries))
	splits = [*ShuffleSplit(tries, test_size=0.3).split(X)]
	for i, mask in enumerate(tqdm(masks)):
		X_masked = X[:,mask]
		for j, (train, test) in enumerate(splits):
			reg = model_fn().fit(X_masked[train], y[train])
			results[i, j] = reg.score(X_masked[test], y[test])
	return results


def results_for_sampler(task, files: list[Path]):
	task_name_date = files[0].parts[-4:-1]
	results_file = cache_path / ("-".join(task_name_date) + '.npy')
	try:
		return np.load(results_file)
	except FileNotFoundError:
		print('Calculating', '/'.join(task_name_date))
		X, y = get_labeled_data(task, files)
		masks = [*map(feature_mask, combinations)]
		results = run_experiment(X, y, masks, 1000, LinearRegression)
		np.save(results_file, results)
		return results

def results_all_samplers(task):
	samplers, files_task = zip(*get_files(task))
	results = np.array(
		[results_for_sampler(task, files_sampler) for files_sampler in files_task])
	return results, samplers
