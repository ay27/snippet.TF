#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ay27 at 2018/4/8
import tensorflow as tf
import os
import numpy as np


class Average(object):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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
