from pathlib import Path
import json

from matplotlib import pyplot as plt
import numpy as np

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

def annotated_heatmap(data, *, title, x_labels, y_labels):
	# adapted from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
	fig, ax = plt.subplots()
	im = ax.imshow(data)

	# Show all ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
	ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(y_labels)):
		for j in range(len(x_labels)):
			text = ax.text(j, i, str(round(data[i, j], 3)),
						ha="center", va="center", color="w")

	ax.set_title(title)
	fig.tight_layout()
	return fig, ax
