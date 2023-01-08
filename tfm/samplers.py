from abc import ABC, abstractmethod
import logging
import math
from typing import Union  # fix when back to 3.10

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import ReLU
from sklearn.cluster import kmeans_plusplus

import utils


def after_activation_function(layers):
    layers = layers.copy()
    remove_ix = [max(i - 1, 0) for i, l in enumerate(layers) if isinstance(l, ReLU)]
    for ix in remove_ix[::-1]:
        del layers[ix]
    return layers


class Sampler(ABC):
    _flat = True

    def __init__(self, n: int):
        self.n = n

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def build(self, layers: list[tf.keras.layers.Layer]):
        self.activations: list[Union[tf.Tensor, np.ndarray]] = []

    @abstractmethod
    def layer(self, x: tf.Tensor, i: int):
        pass

    def accumulate(self) -> np.ndarray:
        return np.hstack(self.activations)

    def apply(self, model: tf.keras.Sequential, included_layers, x: tf.Tensor) -> np.ndarray:
        included_layers = after_activation_function(included_layers)
        self.build(included_layers)
        i = 0
        for layer in model.layers:
            x = layer(x)  # type: ignore
            if layer in included_layers:
                self.layer(tf.reshape(x, (x.shape[0], -1)) if self._flat else x, i)
                i += 1
        return self.accumulate().T


class Random(Sampler):
    def build(self, layers):
        self.sizes = [np.prod(l.output.shape[1:])  # type: ignore
                      for l in layers]
        self.layer_ix = np.r_[0, self.sizes].cumsum()[:-1]
        self.sample_ix = np.sort(np.random.choice(
            sum(self.sizes), self.n, replace=False))  # type: ignore
        super().build(layers)

    def layer(self, x, i):
        ix = self.sample_ix - self.layer_ix[i]
        ix = ix[(ix >= 0) & (ix < self.sizes[i])]
        self.activations += [tf.gather(x, ix, axis=1)]


class MaxImportance(Random):
    def build(self, layers):
        self.top = tf.zeros(0)
        super().build(layers)

    def layer(self, x, i):
        x_abs = tf.math.abs(x)
        augmented = tf.concat([self.top, x_abs], axis=1) if len(self.top) else x_abs
        top_ix, _ = tf.unique(tf.math.argmax(augmented, axis=1))
        self.top = tf.gather(augmented, top_ix, axis=1)
        super().layer(x, i)

    def accumulate(self):
        assert len(self.top) > 0
        logging.debug(f"Part importance vectors: {self.top.shape[1]}")
        random_ix = np.random.choice(
            self.n, self.n - self.top.shape[1], replace=False)
        return np.hstack([self.top, super().accumulate()[:, random_ix]])


class AvgImportance(Sampler):
    @staticmethod
    def score(x: tf.Tensor):
        return tf.math.reduce_mean(x, axis=0)

    def build(self, layers):
        from constants import DATA_SAMPLE_SIZE # fix?
        self.activations = [tf.zeros((DATA_SAMPLE_SIZE, self.n))]
        self.layer_ix = np.full(self.n, -1)
        self.n_layers = len(layers)


    def layer(self, x, i):
        x_abs = tf.math.abs(x)
        augmented = tf.concat([self.activations[0], x_abs], axis=1)
        scores = type(self).score(augmented)
        top_ix = tf.math.top_k(scores, self.n).indices
        logging.debug(f'Average score in layer {i}: {np.mean(scores)}')
        self.layer_ix[top_ix >= self.n] = i
        self.activations[0] = tf.gather(augmented, top_ix, axis=1)


    def accumulate(self) -> np.ndarray:
        counts = [sum(self.layer_ix == i) for i in range(self.n_layers)]
        logging.debug(f'Layer example counts: {counts}')
        return super().accumulate()


class ZeroImportance(AvgImportance):
    # Importance according to number of nonzero elements
    
    @staticmethod
    def score(x: tf.Tensor):
        return -tf.math.count_nonzero(x, axis=0)


class StratifiedSampler(Sampler):
    def build(self, layers):
        sizes = [np.prod(l.output.shape[1:]) for l in layers]  # type: ignore
        self.n_layered = np.zeros_like(layers, dtype=int)
        n_left = self.n
        for i in range(len(layers)):
            self.n_layered[-i] = min(n_left/(len(layers)-i), sizes[-i])
            n_left -= self.n_layered[-i]
        super().build(layers)


class StratifiedRandom(StratifiedSampler):
    def layer(self, x: tf.Tensor, i: int):
        ix = np.random.choice(x.shape[1], self.n_layered[i])
        self.activations += [tf.gather(x, ix, axis=1)]


class StratifiedFilterCorr(StratifiedRandom):
    _flat = False

    def build(self, layers):
        self._layers = layers
        super().build(layers)

    def layer(self, x: tf.Tensor, i: int):
        if (isinstance(self._layers[i], tf.keras.layers.Conv2D) and
                x.shape[1] * x.shape[2] > 1):
            corr = 1 - tf.math.abs(tfp.stats.correlation(x[0], sample_axis=[0,1]))
            filter_ix = utils.farthest(corr, 10)
            x = tf.gather(x, filter_ix, axis=-1)
        super().layer(tf.reshape(x, (x.shape[0], -1)), i)


class StratifiedMaxDist(StratifiedSampler):
    def layer(self, x: tf.Tensor, i: int):
        cur_ix = np.random.random_integers(0, x.shape[1])
        ixs = np.full(x.shape[1], fill_value=False)
        ixs[cur_ix] = True
        x_T = tf.transpose(x).numpy()
        for j in range(self.n_layered[i] - 1):
            print(f'Got activation {j+1}/{self.n_layered[i]} in layer '
                f'{i+1}/{len(self.n_layered)}       ', end='\r')
            with np.errstate(invalid='ignore'):
                y = x_T[cur_ix]
                # cov = tf.math.reduce_mean(
                #     (x_T - tf.reshape(tf.math.reduce_mean(x_T, axis=1), (-1, 1)))
                #     * (y - tf.math.reduce_mean(y)
                # ), axis=1).numpy()
                cov = np.mean(
                    (x_T - x_T.mean(axis=1).reshape((-1, 1))) * (y - y.mean()), axis=1)
                sd = x_T.std(axis=1)*y.std()
                corr = np.abs(np.nan_to_num(cov/sd))

            cur_ix = np.argmin(corr[~ixs])
            ixs[cur_ix] = True
        self.activations += [x_T[ixs].T]


class StratifiedKMeans(StratifiedSampler):
    def __init__(self, n: int, max_n_to_cluster: int):
        self.max_size = max_n_to_cluster
        super().__init__(n)

    @property
    def name(self):
        return super().name + str(self.max_size)

    def layer(self, x: tf.Tensor, i: int):
        # size = x.shape[1]
        # num_chunks = math.ceil(size / self.max_size)  # type: ignore
        # size_chunks = size // num_chunks

        # slices = [slice(j*size_chunks, (j+1)*size_chunks) for j in range(num_chunks)]
        # slices[-1] = slice(slices[-1].start, None)

        # ns_clusters = [round(self.n_layered[i] / num_chunks)] * num_chunks
        # ns_clusters[-1] += self.n_layered[i] - sum(ns_clusters)

        # for j, (slc, n_clusters) in enumerate(zip(slices, ns_clusters)):
        #     centers, _ = kmeans_plusplus(x[:, slc].numpy().T, n_clusters)
        #     self.activations += [centers.T]
        #     print(f'Chunk {j+1}/{num_chunks} in layer {i+1}/{len(self.n_layered)}       ', end='\r')

        size = x.shape[1]
        passes = math.ceil(size / self.max_size)  # type: ignore
        for j in range(passes):
            print(
                f'Chunk {j+1}/{passes} in layer {i+1}/{len(self.n_layered)}       ', end='\r')
            round = math.floor if j < passes - 1 else math.ceil  # fix fix fix
            part = round(self.max_size / passes)
            n_clusters = int(self.n_layered[i] / passes)
            centers, _ = kmeans_plusplus(
                x[:, j*part:(j+1)*part].numpy().T,
                n_clusters + int(self.n_layered[i] %
                                 n_clusters if j == passes - 1 else 0)
            )
            self.activations += [centers.T]


# def stratify(sampler: Sampler) -> Sampler:
#     def build(self, layers: list[tf.keras.layers.Layer]):
#         sizes = [np.prod(l.output.shape[1:]) for l in layers]  # type: ignore
#         self.n_layered = np.zeros_like(layers)
#         n_left = self.n
#         for i in range(len(layers)):
#             self.n_layered[-i] = min(n_left/i, sizes[-i])
#             n_left -= self.n_layered[-1]
#         self._model_layers = layers
#         self._activations = []

#     def layer(x: tf.Tensor, i: int):
#         self.n = self.n_layered[i]
#         sampler.build(self, [self._model_layers[i]])
#         sampler.layer(self, x, i)
