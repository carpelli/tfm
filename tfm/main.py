import argparse
from datetime import datetime
import errno
import logging
import os
from pathlib import Path

# import tensorflow as tf

import pd
import utils
from utils.timertree import timer
from constants import OUTDIR, INPUT_DIR, TIMEOUT, SAMPLER, DATA_SAMPLE_SIZE, VERSION


if __name__ == "__main__":
    # tf.random.set_seed(1234)
    # np.random.seed(1234)
    # tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path, help="path to the google_data folder")
    parser.add_argument('--task', choices=[1, 2], type=int, required=True)
    parser.add_argument('-o', '--output', default=OUTDIR, type=Path)
    parser.add_argument('--timeout', default=TIMEOUT, type=int)
    # parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--models', nargs='+', default=[])
    parser.add_argument('--from-dm')
    args = parser.parse_args()

    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stdoutHandler = logging.StreamHandler()
    stdoutHandler.setFormatter(formatter)
    logging.basicConfig(handlers=[stdoutHandler], level=logging.INFO)

    # set the outdir depending on if asked to continue the last batch or make a new one
    outdir = Path(args.output) / f'task{args.task}/{SAMPLER.name}{"-" if VERSION else ""}{VERSION}'
    subdirs = sorted(outdir.glob("[0-9]*"))
    if args.resume:
        if subdirs:
            outdir /= subdirs[-1]
        else:
            raise FileNotFoundError(errno.ENOENT, 'Asked to --resume, but no '
                'directories to resume from')
    elif args.from_dm:
        outdir /= args.from_dm
    else:
        outdir /= datetime.now().strftime('%y.%m.%d-%H.%M.%S')
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / 'log.txt').touch(exist_ok=True) # fix
    fileHandler = logging.FileHandler(outdir / 'log.txt')
    fileHandler.setFormatter(formatter)
    logging.getLogger().addHandler(fileHandler)

    if args.from_dm:
        logging.info(f'Starting calculating PDs for models {args.models} with timeout '
            f'{args.timeout}')

        for path in [outdir / f'model_{m}' for m in args.models]:
            with timer(path.name):
                pd.from_dm_and_save(utils.load('dm', path), path, args.timeout)

        exit()

    logging.info(f'Starting experiment on task {args.task} with {SAMPLER.name} '
        f'(version \'{VERSION}\') sampler timing out after {args.timeout}s')
    logging.info(f"Importing data...")

    input_dir = args.data_path / INPUT_DIR[args.task]
    x_train, entropy = utils.import_and_sample_data(
        input_dir / "dataset_1", DATA_SAMPLE_SIZE)
    logging.info(f"Data sampling entropy: {entropy}")

    logging.info(f"Finished importing data")

    if args.models:
        model_paths = [input_dir / f'model_{m}' for m in args.models]
    else:
        model_paths = input_dir.glob('model*')

    for model_path in sorted(model_paths, key=lambda p: (len(p.name), p.name)):
        pd_path = outdir / f'{model_path.name}'

        if already_present := [*outdir.glob(pd_path.name + "*")]:
            logging.info(f'Found {already_present[0]}, skipping...')
            continue
        
        try:
            model = utils.import_model(model_path)
        except FileNotFoundError:
            logging.warning(model_path.name + ' cannot be found')
            continue
        
        with timer(model_path.name):
            pd.from_model_and_save(model, x_train, pd_path, args.timeout)
