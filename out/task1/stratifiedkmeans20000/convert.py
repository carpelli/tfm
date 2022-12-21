# Generate an std_log so that it may be used for calculation time comparisons

from glob import glob
import json
from pathlib import Path


log = 'Log has been generated for use in PD time calculation, not an actual log!\n\n\n'
# for model in json.load(open(next(glob('timer/*.json')))):

for model, times in json.load(open(next(Path(__file__).parent.glob('timer/*.json')))).items():
	for name, time in times.items():
		if name[0] == '_':
			log += f'timed {model}: {time}s\n'
		else:
			log += f'timed {model}/{name}: {time}s\n'

log += open(Path(__file__).parent / 'timer/shellout.txt').read()
(Path(__file__).parent / 'std_log.txt').write_text(log)
