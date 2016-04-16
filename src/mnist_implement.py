#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNNsのモデルを実行するスクリプト
今回使用するデータセットはMNIST
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mnist_classification import CNN
from chainer import cuda
import numpy as np
from sklearn.datasets import fetch_mldata

#GPUつかうよ
cuda.init(0)

print('load MNIST digit dataset')
mnist = fetch_mldata('MNIST original', data_home=".")
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

output_dim = 10

print('create CNNs model')
cnn = CNN(data=mnist.data,
          target=mnist.target,
          gpu=0,
          output_dim=output_dim)

print('training and test')
cnn.train_and_test(n_epoch=100)

print('end')

