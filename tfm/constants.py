from pathlib import Path

import samplers


DATA_SAMPLE_SIZE = 2000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
INPUT_DIR = {
	1: Path('public_data/input_data/task1_v4'),
	2: Path('public_data/input_data/task2_v1')
}
SAMPLER = samplers.AvgImportance(3000)
TIMEOUT = 1000
