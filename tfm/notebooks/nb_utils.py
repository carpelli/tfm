from pathlib import Path
import json

outdir = Path('../../out/')
ref_file = Path(
    "/Users/otis/Documents/rubens_speelhoekje/google/google_data/public_data/reference_data/task1_v4/model_configs.json")

gen_gaps = {
	'model_' + k: v['metrics']['train_acc'] - v['metrics']['test_acc']
	for k, v in json.loads(ref_file.read_text()).items()
}

def dirs(path: Path):
	for dir in path.iterdir():
		if dir.is_dir():
			yield dir
