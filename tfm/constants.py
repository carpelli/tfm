from pathlib import Path

import samplers


DATA_SAMPLE_SIZE = 2000
NEURON_SAMPLE_SIZE = 3000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
INPUT_DIR = Path('public_data/input_data')
SAMPLER = samplers.StratifiedKMeans(NEURON_SAMPLE_SIZE, 20000)
VERSION = ''
INCLUDED_LAYERS = {
	'': slice(1, None),
	'WithFirst': slice(0, None)
}
TIMEOUT = 1000
