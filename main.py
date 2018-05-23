#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/21
import time

import tensorflow as tf
import numpy as np
import utils
from logger import Logger, CSVLogger
from reader import read_from_numpy
from model import LrModel
import os
from tqdm import tqdm
import config

utils.set_best_gpu(config.num_gpus)
utils.set_seed(config.seed)

flags = tf.flags
flags.DEFINE_string("model_save_path", "./save/", "model save path")
flags.DEFINE_string("log_path", "./log/", "log files save path")
FLAGS = flags.FLAGS

utils.check_or_create(FLAGS.model_save_path)
utils.check_or_create(FLAGS.log_path)


def run_epoch(sess, loss_op, iterations, eval_op=None):
    loss = utils.MovingMean()
    fetch_ops = {
        "loss": loss_op,
    }
    if eval_op is not None:
        fetch_ops["eval_op"] = eval_op

    qm = tqdm(range(iterations), ascii=True, ncols=80)
    for _ in qm:
        vals = sess.run(fetch_ops)
        loss.move(vals["loss"])
        qm.set_description("Loss=%.3f" % loss)
    return float(loss)


def main(_):
    # enlarge batch size to match multi gpus
    train_X, train_Y, train_iters = read_from_numpy(config.epochs, config.batch_size * config.num_gpus)
    test_X, test_Y, test_iters = read_from_numpy(config.epochs, config.batch_size)

    lr = tf.Variable(config.lr, trainable=False)
    lr = tf.train.exponential_decay(lr,
                                    tf.train.get_or_create_global_step(),
                                    config.lr_decay_interval * train_iters,
                                    config.lr_decay,
                                    staircase=True, name="lr")
    optimizer = tf.train.GradientDescentOptimizer(lr, name="GradientDescent")

    loss_op, train_op = utils.parallel(train_X, train_Y, config.num_gpus,
                                       optimizer,
                                       lambda x, y: LrModel(x, y, is_training=True, reuse=False))

    # train_model = LrModel(train_X, train_Y, lr=config.lr, is_training=True, reuse=False)
    test_model = LrModel(test_X, test_Y, is_training=False, reuse=True)

    logger = Logger(FLAGS.log_path)
    csv_logger = CSVLogger(os.path.join(FLAGS.log_path, "log.csv"), ["epoch", "Lr", "Train Loss", "Valid Loss"])

    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session(config=utils.gen_session_config())

    sess.run(init)

    for epoch in range(config.epochs):
        train_loss = run_epoch(sess, loss_op, train_iters, train_op)
        valid_loss = run_epoch(sess, test_model.loss, test_iters)
        lr_val = sess.run(lr)
        csv_logger.log(epoch, lr_val, train_loss, valid_loss)
        logger.scalar_summary("Train Loss", train_loss, epoch)
        logger.scalar_summary("Valid Loss", valid_loss, epoch)
        logger.scalar_summary("Lr", lr_val, epoch)

        print("Epoch:%d Lr: %.6f Train Loss = %.3f, Valid Loss = %.3f" % (epoch, lr_val, train_loss, valid_loss))
        print("=" * 90)
    saver.save(sess, os.path.join(FLAGS.model_save_path, "model.ckpt"))


if __name__ == '__main__':
    tf.app.run()
