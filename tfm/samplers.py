from functools import partial
from itertools import islice, compress
import math
from types import NoneType
from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import tensorflow as tf


def outputs(model, x, skip_n=1, exclude=NoneType) -> Generator[tf.Tensor, None, None]:
    outputs = (x := l(x) for l in model.layers)
    mask = (not isinstance(l, exclude) for l in model.layers)
    exclude_layers = compress(outputs, mask)
    skip_first = islice(exclude_layers, skip_n, None)
    return (tf.reshape(x, (x.shape[0], -1)) for x in skip_first)


def random(n):
    def sampler(model, x):
        sizes = [np.prod(l.output.shape[1:]) for l in model.layers[1:]]
        sample_idx = np.sort(np.random.choice(sum(sizes), n, replace=False))  # type: ignore
        bin_sizes = np.histogram(sample_idx, np.r_[0, sizes].cumsum())[0]
        sample_idx_bins = np.split(
            sample_idx % np.repeat(sizes, bin_sizes),
            bin_sizes[:-1].cumsum())
        return np.vstack(
            [y.T[idx] for y, idx in zip(outputs(model, x), sample_idx_bins)])
    return sampler

def importance(max_samples_from_hidden_layers=3000, num_skipped_layers_from_start=1):
    def _examples_x_activations_for_input_importance_sampling(model, x):
        '''
        We take for each non-skipped hidden layer the max(number_of_neurons_layer, number_of_samples) neurons with highest
        mean activation value in absolute value across all the dataset x where number_of_samples is equal to
        math.floor(max_samples_from_hidden_layers/raw_total_layers) for all the non-first hidden layers and
        math.floor(max_samples_from_hidden_layers/raw_total_layers) + max_samples_from_hidden_layers%raw_total_layers
        for the first hidden layer. For the output layer, we add all its neurons.
        :param x: dataset
        :param model: neural network in tensorflow keras.
        :param num_skipped_layers_from_start: Number of layers to skip from the start to do the analysis
        :param max_samples_from_hidden_layers: max number of elements in the sample from the hidden layers.
        :return:
        '''
        # We add always the last layer completely
        raw_total_layers = sum([not isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers])
        # We remove the last layer (the -1) because we add it completely. The first one is not counted.
        number_of_hidden_layers = raw_total_layers - num_skipped_layers_from_start - 1
        sampled_activations_bd = None  # final shape=(total_sampled_neurons, number_of_examples)
        samples_per_ordinary_layer = math.floor(max_samples_from_hidden_layers / number_of_hidden_layers)
        excess_of_neurons = max_samples_from_hidden_layers % number_of_hidden_layers
        skipped_iterations = 0
        layer_idx = -1
        for layer in model.layers:
            if skipped_iterations < num_skipped_layers_from_start:
                x = layer(x)
                if not isinstance(layer, tf.keras.layers.Dropout): # We only count non-dropout layers
                    skipped_iterations += 1
            else:
                x = layer(x)
                if not isinstance(layer, tf.keras.layers.Dropout):
                    layer_idx += 1  # We start with the layer 0 when we have the first layer that is not dropout
                    examples_x_neurons = tf.reshape(x, (x.shape[0], -1))
                    if layer_idx < number_of_hidden_layers:
                        averages_x_neurons = tf.reduce_mean(tf.math.abs(examples_x_neurons), axis=0)
                        averages_x_neurons_args_sorted = tf.argsort(averages_x_neurons, axis=0, direction='DESCENDING')
                        number_of_samples = samples_per_ordinary_layer + excess_of_neurons if layer_idx == 0 \
                            else samples_per_ordinary_layer
                        selected_examples_x_neurons = tf.gather(examples_x_neurons,
                                                                indices=averages_x_neurons_args_sorted[:number_of_samples],
                                                                axis=1)

                        sampled_activations_bd = selected_examples_x_neurons if sampled_activations_bd is None else \
                            tf.concat((sampled_activations_bd, selected_examples_x_neurons), axis=1)
                    else:
                        # We are in the last layer (output layer). We add all the neurons
                        sampled_activations_bd = examples_x_neurons if sampled_activations_bd is None else \
                            tf.concat((sampled_activations_bd, examples_x_neurons), axis=1)
        return sampled_activations_bd
    return _examples_x_activations_for_input_importance_sampling
