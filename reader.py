#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/21
import tensorflow as tf
import numpy as np

W = np.random.rand(20, 100) * 10.0


def read_from_numpy(num_epochs, batch_size, name=None):
    # refer to https://www.tensorflow.org/programmers_guide/datasets for more detail!

    num_examples = 1000
    X = np.random.rand(num_examples, 20) * 10.0

    Y = np.matmul(X, W)
    with tf.variable_scope(name, "Reader-Npy"):
        # another choice is use placeholder to reduce the graph size
        # x_placeholder = tf.placeholder(X.dtype, X.shape, name="X-pholder")
        # y_placeholder = tf.placeholder(Y.dtype, Y.shape, name="Y-pholder")

        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X, dtype=tf.float32), tf.constant(Y, dtype=tf.float32)))
        # preprocess
        dataset = dataset.map(
            lambda x, y: (x + tf.random_uniform(x.get_shape(), minval=0, maxval=0.02, dtype=tf.float32), y),
            num_parallel_calls=4)
        dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(buffer_size=1000)
        iterator = dataset.make_one_shot_iterator()

        x_batch, y_batch = iterator.get_next()

    return x_batch, y_batch, num_examples // batch_size + (0 if num_examples % batch_size == 0 else 1)


def read_from_generator(num_epochs, batch_size, name=None):
    # refer to https://www.tensorflow.org/programmers_guide/datasets for more detail!

    num_examples = 1000
    X = np.random.rand(num_examples, 20) * 10.0

    Y = np.matmul(X, W)

    def generator():
        for ii in range(num_examples):
            yield X[ii], Y[ii]

    with tf.variable_scope(name, "Reader-Npy"):
        dataset = tf.data.Dataset.from_generator(generator,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([20]), tf.TensorShape([100])))
        # preprocess
        dataset = dataset.map(
            lambda x, y: (x + tf.random_uniform(tf.shape(x), minval=0, maxval=0.02, dtype=tf.float32), y),
            num_parallel_calls=4)
        dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(buffer_size=1000)
        iterator = dataset.make_one_shot_iterator()

        x_batch, y_batch = iterator.get_next()

    return x_batch, y_batch, num_examples // batch_size + (0 if num_examples % batch_size == 0 else 1)
