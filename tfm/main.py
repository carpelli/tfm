import argparse
from datetime import datetime
import logging
from pathlib import Path

import tensorflow as tf

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
    args = parser.parse_args()

    if args.interactive:
        ask_for_args(args)

    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    stdoutHandler = logging.StreamHandler()
    stdoutHandler.setFormatter(formatter)
    stdoutHandler.setLevel(logging.INFO)
    logging.basicConfig(handlers=[stdoutHandler], level=logging.INFO)
    # logging.getLogger()

    # set the outdir depending on if asked to continue the last batch or make a new one
    outdir = Path(args.output) / 'task1' / SAMPLER.name
    subdirs = sorted(outdir.glob("[0-9]"))
    if args.resume:
        if subdirs:
            outdir /= subdirs[-1]
        else:
            logging.warning("Asked to resume but no previous output found, \
                making a new directory")
    else:
        outdir /= datetime.now().strftime('%y.%m.%d-%H')
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / 'log.txt').touch(exist_ok=True) # fix
    fileHandler = logging.FileHandler(outdir / 'log.txt')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fileHandler)

    logging.info(
        f"Starting experiment with {SAMPLER.name} timing out after {args.timeout}s")
    logging.info(f"Importing data...")

    x_train = utils.import_and_sample_data(
        args.data_path / "dataset_1", DATA_SAMPLE_SIZE)

    logging.info(f"Finished importing data")

    # sort the models according to their number
    model_paths = sorted(args.data_path.glob('model*'),
                         key=lambda p: (len(p.name), p.name))

    for model_path in model_paths:
        pd_path = outdir / f'{model_path.name}'

        try:
            file = next(outdir.glob(pd_path.name + "*"))
            logging.info(f'Found {file}, skipping...')
            continue
        except StopIteration:
            pass

        model = utils.import_model(model_path)
        with timer(model_path.name):
            pd.from_model_and_save(model, x_train, pd_path, args.timeout)
