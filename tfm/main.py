import argparse
from datetime import datetime
import logging
from pathlib import Path

# import tensorflow as tf

import pd
import utils
from utils.timertree import timer
from constants import OUTDIR, TIMEOUT, SAMPLER, DATA_SAMPLE_SIZE


def ask_for_args(args):
    for arg, value in args.__dict__.items():
        if value == parser.get_default(arg):
            if type(value) == bool:
                args.__dict__[arg] = \
                    input(f'{arg.title()} (n)? y for yes: ') == 'y'
            elif type(value) == list:
                response = input(f'{arg.title()}: ')
                args.__dict__[arg] = response.split(' ') if response else []
            else:
                args.__dict__[arg] = type(value)(
                    input(f'{arg.title()} ({value}): ') or value)

if __name__ == "__main__":
    # tf.random.set_seed(1234)
    # np.random.seed(1234)
    # tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=Path)
    parser.add_argument('-o', '--output', default=OUTDIR, type=Path)
    parser.add_argument('--timeout', default=TIMEOUT, type=int)
    # parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--models', nargs='+', default=[])
    args = parser.parse_args()

    parser._get_args

    if args.interactive:
        ask_for_args(args)

    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stdoutHandler = logging.StreamHandler()
    stdoutHandler.setFormatter(formatter)
    logging.basicConfig(handlers=[stdoutHandler], level=logging.INFO)
    # logging.getLogger()

    # set the outdir depending on if asked to continue the last batch or make a new one
    outdir = Path(args.output) / 'task1' / SAMPLER.name
    subdirs = sorted(outdir.glob("[0-9]"))
    if args.resume:
        if subdirs:
            outdir /= subdirs[-1]
        else:
            logging.warning("Asked to resume but no previous output found. \
                Making a new directory")
    else:
        outdir /= datetime.now().strftime('%y.%m.%d-%H.%M.%S')
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / 'log.txt').touch(exist_ok=True) # fix
    fileHandler = logging.FileHandler(outdir / 'log.txt')
    fileHandler.setFormatter(formatter)
    logging.getLogger().addHandler(fileHandler)

    logging.info(
        f"Starting experiment with {SAMPLER.name} timing out after {args.timeout}s")
    logging.info(f"Importing data...")

    x_train = utils.import_and_sample_data(
        args.data_path / "dataset_1", DATA_SAMPLE_SIZE)

    logging.info(f"Finished importing data")

    if args.models:
        model_paths = [args.data_path / f'model_{m}' for m in args.models]
    else:
        model_paths = args.data_path.glob('model*')

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
