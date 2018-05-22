#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/5/21
import tensorflow as tf
import numpy as np
import utils
from logger import Logger, CSVLogger
from reader import read_from_numpy
from model import LrModel
import os
from tqdm import tqdm
import config

utils.set_best_gpu(1)
utils.set_seed(config.seed)

flags = tf.flags
flags.DEFINE_string("model_save_path", "./save/", "model save path")
flags.DEFINE_string("log_path", "./log/", "log files save path")
FLAGS = flags.FLAGS

utils.check_or_create(FLAGS.model_save_path)
utils.check_or_create(FLAGS.log_path)


def run_epoch(sess, model, iterations, eval_op=None):
    loss = utils.MovingMean()
    fetch_ops = {
        "loss": model.loss,
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
    train_X, train_Y, iterations = read_from_numpy(config.epochs, config.batch_size)
    test_X, test_Y, iterations = read_from_numpy(config.epochs, config.batch_size)

    lr = tf.Variable(config.lr, trainable=False)
    lr = tf.train.exponential_decay(lr, tf.train.get_or_create_global_step(), config.lr_decay_interval * iterations,
                                    config.lr_decay,
                                    staircase=True, name="lr")
    train_model = LrModel(train_X, train_Y, lr=config.lr, is_training=True, reuse=False)
    test_model = LrModel(test_X, test_Y, is_training=False, reuse=True)

    logger = Logger(FLAGS.log_path)
    csv_logger = CSVLogger(os.path.join(FLAGS.log_path, "log.csv"), ["epoch", "Lr", "Train Loss", "Valid Loss"])

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session(config=utils.gen_session_config())

    sess.run(init)

    for epoch in range(config.epochs):
        train_loss = run_epoch(sess, train_model, iterations, train_model.train_op)
        valid_loss = run_epoch(sess, test_model, iterations)
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
