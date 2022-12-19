from abc import ABC, abstractmethod
import logging
import math
from typing import Union  # fix when back to 3.10

import numpy as np
import tensorflow as tf

from sklearn.cluster import kmeans_plusplus


class Sampler(ABC):
    def __init__(self, n: int):
        self.n = n

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @abstractmethod
    def build(self, layers: list[tf.keras.layers.Layer]):
        self.activations: list[Union[tf.Tensor, np.ndarray]] = []

    @abstractmethod
    def layer(self, x: tf.Tensor, i: int):
        pass

    def accumulate(self) -> np.ndarray:
        return np.hstack(self.activations)


class Random(Sampler):
    def build(self, layers):
        self.sizes = [np.prod(l.output.shape[1:])
                      for l in layers]  # type: ignore
        self.layer_ix = np.r_[0, self.sizes].cumsum()[:-1]
        self.sample_ix = np.sort(np.random.choice(
            sum(self.sizes), self.n, replace=False))  # type: ignore
        super().build(layers)

    def layer(self, x, i):
        ix = self.sample_ix - self.layer_ix[i]
        ix = ix[(ix >= 0) & (ix < self.sizes[i])]
        self.activations += [tf.gather(x, ix, axis=1)]


class Importance(Random):
    def build(self, layers):
        self.top = tf.zeros(0)
        super().build(layers)

    def layer(self, x, i):
        augmented = tf.concat([self.top, x], axis=1) if len(self.top) else x
        top_ix, _ = tf.unique(tf.math.argmax(augmented, axis=1))
        self.top = tf.gather(augmented, top_ix, axis=1)
        super().layer(x, i)

    def accumulate(self):
        assert len(self.top) > 0
        logging.debug(f"Part importance vectors: {self.top.shape[1]}")
        random_ix = np.random.choice(
            self.n, self.n - self.top.shape[1], replace=False)
        return np.hstack([self.top, super().accumulate()[:, random_ix]])


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


class StratifiedKMeans(StratifiedSampler):
    def __init__(self, n: int, max_n_to_cluster: int):
        self.max_size = max_n_to_cluster
        super().__init__(n)

    @property
    def name(self):
        return super().name + str(self.max_size)

    def layer(self, x: tf.Tensor, i: int):
        size = x.shape[1]
        num_chunks = math.ceil(size / self.max_size)  # type: ignore
        size_chunks = size // num_chunks

        slices = [slice(j*size_chunks, (j+1)*size_chunks) for j in range(num_chunks)]
        slices[-1] = slice(slices[-1].start, None)

        ns_clusters = [round(self.n_layered[i] / num_chunks)] * num_chunks
        ns_clusters[-1] += self.n_layered[i] - sum(ns_clusters)

        for j, (slc, n_clusters) in enumerate(zip(slices, ns_clusters)):
            centers, _ = kmeans_plusplus(x[:, slc].numpy().T, n_clusters)
            self.activations += [centers.T]
            print(f'Chunk {j+1}/{num_chunks} in layer {i+1}/{len(self.n_layered)}       ', end='\r')


def apply(sampler: Sampler, model: tf.keras.Sequential, included_layers, x: tf.Tensor) -> np.ndarray:
    sampler.build(included_layers)
    i = 0
    for layer in model.layers:
        x = layer(x)  # type: ignore
        if layer in included_layers:
            sampler.layer(tf.reshape(x, (x.shape[0], -1)), i)
            i += 1
    return sampler.accumulate().T


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
