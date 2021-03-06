#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/22
import tensorflow as tf
from reader import read_from_numpy, read_from_generator
from model import LrModel
import utils

utils.set_best_gpu(1)
utils.set_seed(1234)


def run_epoch(sess, model, iterations):
    loss = utils.MovingMean()
    for _ in range(iterations):
        loss_val = sess.run(model.loss)
        loss.move(loss_val)
    return loss


def main1(_):
    # another choice to restore the model without re-creating it.
    # should be noted that if we enable multi gpus training,
    # we must enable `allow_soft_placement` to restore the model properly.
    sess = tf.Session(config=utils.gen_session_config(allow_soft_placement=True))
    saver = tf.train.import_meta_graph("./save/model.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./save/"))

    W = tf.get_default_graph().get_tensor_by_name("LR-Model/Weight:0")
    W = sess.run(W)
    print(W.shape)


def main(_):
    epochs = 1
    batch_size = 32
    X, Y, iterations = read_from_generator(epochs, batch_size)
    test_model = LrModel(X, Y, is_training=False, reuse=False)

    saver = tf.train.Saver()
    sess = tf.Session(config=utils.gen_session_config())

    ckpt = tf.train.get_checkpoint_state("./save/")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # restore the whole model and test it
    print(run_epoch(sess, test_model, iterations))


if __name__ == '__main__':
    tf.app.run(main1)
