from pathlib import Path

import samplers


DATA_SAMPLE_SIZE = 2000
UPPER_DIM = 1
OUTDIR = Path(__file__).parent / '../out'
SAMPLER = samplers.AvgImportance(3000)
TIMEOUT = 1000
