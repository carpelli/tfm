from itertools import accumulate, islice

import numpy as np


def outputs(model, x, skipfirst=1):
    return map(
        lambda x: np.reshape(x, (x.shape[0], -1)),
        islice(
            accumulate(model.layers, lambda x, l: l(x), initial=x),
            skipfirst + 1, None))


def random(n):
    def sampler(model, x):
        sizes = [np.prod(l.output.shape[1:]) for l in model.layers[1:]]
        sample_idx = np.sort(np.random.choice(sum(sizes), n, replace=False))
        bin_sizes = np.histogram(sample_idx, np.r_[0, sizes].cumsum())[0]
        sample_idx_bins = np.split(
            sample_idx % np.repeat(sizes, bin_sizes),
            bin_sizes[:-1].cumsum())
        return np.vstack(
            [y.T[idx] for y, idx in zip(outputs(model, x), sample_idx_bins)])
    return sampler
