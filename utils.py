#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/4/8
import sys
import time

import tensorflow as tf
import os
import numpy as np


class Average(object):
    """
    Full Average without saving.
    """

    def __init__(self):
        self._val = 0.0
        self._iter = 0

    def move(self, val):
        self._val = (self._val * self._iter + val) / (self._iter + 1)
        self._iter += 1
        return self._val

    def __float__(self):
        return self._val

    def __str__(self):
        return str(self._val)


class MovingMean(object):
    def __init__(self, N=100):
        """
        Moving Mean (Average) for a fix length N.

        Parameters
        ----------
        N: int
            moving length
        """
        self._N = N
        self._log = np.zeros(self._N, dtype=np.float32)
        self._idx = 0
        self._val = 0.0
        self._first_round = True

    def move(self, val):
        self._log[self._idx] = val
        self._idx = (self._idx + 1) % self._N
        if self._idx == 0:
            self._first_round = False
        if self._first_round:
            self._val = np.mean(self._log[:self._idx])
        else:
            self._val = np.mean(self._log)
        return self._val

    def __float__(self):
        return float(self._val)

    def __str__(self):
        return str(self._val)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ShowProcess(object):
    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        self.start_ts = None

    def show_process(self, i=None, msg=''):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        if self.start_ts is None:
            self.start_ts = time.time()
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + '] '\
                      + '%.2f' % percent + '% | ' + '%.1fs | ' % (time.time() - self.start_ts) + msg + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self, words=None):
        if words:
            print(words)
        self.i = 0
        self.start_ts = None


def get_valid_save_path(save_path, prefix=None):
    ii = 0
    if prefix:
        tmp = os.path.join(save_path, prefix)
    else:
        tmp = save_path
    while True:
        if os.path.exists('%s-%d' % (tmp, ii)):
            ii += 1
        else:
            res = '%s-%d' % (tmp, ii)
            os.mkdir(res)
            print('mkdir ', res)
            break
    return res


class Schedule(object):
    def __init__(self):
        self.interval_step = 0
        self.tasks = []
        self.interval = []

    def add(self, task_func, interval):
        self.tasks.append(task_func)
        self.interval.append(interval)
        return self

    def ticktock(self):
        self.interval_step += 1
        for t, val in zip(self.tasks, self.interval):
            if (isinstance(val, list) and self.interval_step in val) \
                    or (self.interval_step % val == 0):
                t()


class VariableContainer(object):
    def __init__(self, init_value, dtype, name, reuse=False, scope=None):
        with tf.variable_scope(scope, "VariableContainer", reuse=reuse):
            self._var = tf.get_variable(name, dtype=dtype, initializer=tf.constant(init_value), trainable=False)
        self._value = init_value
        self._new_value = tf.placeholder(dtype, shape=[], name="new_%s" % name)
        self._assign_op = tf.assign(self._var, self._new_value)

    @property
    def data(self):
        return self._value

    @property
    def var(self):
        return self._var

    def __int__(self):
        if isinstance(self._value, int):
            return self._value
        else:
            raise TypeError

    def __float__(self):
        if isinstance(self._value, float):
            return self._value
        else:
            raise TypeError

    def decay(self, sess, decay_rate):
        self._value = self._value * decay_rate
        sess.run(self._assign_op, feed_dict={self._new_value: self._value})

    def set(self, sess, value):
        self._value = value
        sess.run(self._assign_op, feed_dict={self._new_value: self._value})


def _average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def parallel(X, Y, num_gpus, optimizer, build_model_func):
    """

    Parameters
    ----------
    X
    Y
    num_gpus
    optimizer
    build_model_func:
    a function like `func(x, y)`. used to create model in each gpus

    Returns
    -------
    loss_op, train_op

    """
    Xs = tf.split(X, num_gpus)
    Ys = tf.split(Y, num_gpus)

    tower_grads = []
    tower_loss = []
    for d in range(num_gpus):
        with tf.device('/gpu:%s' % d):
            with tf.name_scope('%s_%s' % ('tower', d)):
                model = build_model_func(Xs[d], Ys[d])
                with tf.variable_scope("loss"):
                    grads = optimizer.compute_gradients(model.loss)
                    tower_grads.append(grads)
                    tower_loss.append(model.loss)
                    tf.get_variable_scope().reuse_variables()
                tf.get_variable_scope().reuse_variables()

    mean_loss = tf.stack(axis=0, values=tower_loss)
    mean_loss = tf.reduce_mean(mean_loss, axis=0)
    mean_grads = _average_gradients(tower_grads)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(mean_grads, global_step=tf.train.get_or_create_global_step())
    return mean_loss, train_op


def check_or_create(path):
    if not os.path.isdir(path) and not os.path.isfile(path):
        os.mkdir(path)


def set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def gen_session_config(gpu_memory_ratio=-1, allow_soft_placement=False):
    if gpu_memory_ratio == -1:
        config = tf.ConfigProto()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_ratio)
        config = tf.ConfigProto(gpu_options=gpu_options)

    config.gpu_options.allow_growth = True
    config.allow_soft_placement = allow_soft_placement
    return config


def set_best_gpu(top_k=1):
    best_gpu = _scan(top_k)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, best_gpu))
    print('CUDA_VISIBLE_DEVICES: ', os.environ["CUDA_VISIBLE_DEVICES"])
    return best_gpu


def _scan(top_k):
    CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
    CMD2 = 'nvidia-smi -L | wc -l'
    CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

    total_gpu = int(os.popen(CMD2).read())

    assert top_k <= total_gpu, 'top_k > total_gpu !'

    # first choose the free gpus
    gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
    free_gpus = set(range(total_gpu)) - gpu_usage

    # then choose the most memory free gpus
    gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
    gpu_sorted = list(sorted(range(total_gpu), key=lambda x: gpu_free_mem[x], reverse=True))[len(free_gpus):]

    res = list(free_gpus) + list(gpu_sorted)
    return res[:top_k]


if __name__ == '__main__':
    print(set_best_gpu(1))
    x = Average()

    for ii in range(2000):
        x.move(ii)
    print(x)
    print(np.mean(np.arange(2000)))
