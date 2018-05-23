#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/21
import tensorflow as tf


class LrModel(object):
    def __init__(self, inputs, labels, is_training=True, reuse=False):
        self._inputs = inputs
        self._labels = labels
        self.is_training = is_training
        self._reuse = reuse
        self._def_model()

    def _def_model(self):
        with tf.variable_scope("LR-Model", values=[self._inputs, self._labels], reuse=self._reuse):
            self._W = tf.get_variable("Weight",
                                      (self._inputs.shape[1], self._labels.shape[1]),
                                      dtype=tf.float32,
                                      trainable=self.is_training)
            self._b = tf.get_variable("bias",
                                      (self._labels.shape[1]),
                                      dtype=tf.float32,
                                      trainable=self.is_training)
            self._logits = tf.nn.xw_plus_b(self._inputs, self._W, self._b, name="xw_plus_b")
            self.loss = tf.nn.l2_loss(self._labels - self._logits, name="loss")

