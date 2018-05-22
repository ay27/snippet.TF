#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/22
import tensorflow as tf
from reader import read_from_numpy, read_from_generator
from model import LrModel
import utils
utils.set_best_gpu()
utils.set_seed(1234)


def run_epoch(sess, model, iterations, eval_op=None):
    loss = utils.MovingMean()
    fetch_ops = {
        "loss": model.loss,
    }
    if eval_op is not None:
        fetch_ops["eval_op"] = eval_op
    for _ in range(iterations):
        vals = sess.run(fetch_ops)
        loss.move(vals["loss"])
    return loss


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

    # or we can obtain some special tensor to process
    W = tf.get_default_graph().get_tensor_by_name("LR-Model/Weight:0")
    W = sess.run(W)
    print(W.shape)


if __name__ == '__main__':
    tf.app.run()
