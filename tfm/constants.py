from pathlib import Path

import samplers


DATA_SAMPLE_SIZE = 2000
NEURON_SAMPLE_SIZE = 3000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
INPUT_DIR = {
	1: Path('public_data/input_data/task1_v4'),
	2: Path('public_data/input_data/task2_v1')
}
SAMPLER = samplers.AvgImportance(NEURON_SAMPLE_SIZE)
VERSION = ''
INCLUDED_LAYERS = {
	'': slice(1, None),
	'WithFirst': slice(0, None)
}
TIMEOUT = 1000
