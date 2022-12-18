from pathlib import Path
import pickle

import numpy as np

import utils
from ruben import PersistenceDiagram


def extract_pd(path: Path):
	with open(path, 'rb') as f:
		obj: PersistenceDiagram = pickle.load(f)
	return np.array([[p.birth, p.death, p.dim] for p in obj.points])


for file in Path('out').glob('**/*.bin'):
	utils.save(extract_pd(file), 'pd', file.with_suffix(''))
	file.unlink()
