#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNNsのモデルを実行するスクリプト
今回使用するデータセットはMNIST
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from get_image_tensor import Input_data
from deeper_classification import CNN
from chainer import cuda
import numpy as np
from sklearn.datasets import fetch_mldata

#GPUつかうよ
cuda.init(0)

print('load dataset')
###
output_dim = 2
data_list= "/home/yuma/programing/github/ikemen_check/target/bijo_target.csv"
input_data = Input_data(data_list)
###

print('create CNNs model')
cnn = CNN(data=input_data.data,
          target=input_data.target,
          gpu=0,
          output_dim=output_dim)

print('training and test')
cnn.train_and_test(n_epoch=100, batchsize=20)
print('end')

