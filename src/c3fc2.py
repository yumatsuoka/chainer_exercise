#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ニューラルネットライブラリchainerを用いて
CNNを作成する。データセットはMNISTで動作を確かめる
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six.moves.cPickle as pickle
from sklearn.cross_validation import train_test_split
#import pylab
#import matplotlib.pyplot as plt
import chainer.functions as F
from chainer import cuda, Variable, FunctionSet, optimizers
import sys

class C3fc2(FunctionSet):
    """CNNのモデルの１つ"""
    def __init__(self, output_dim):
        super(C3fc2, self).__init__(
                #Convolution2D(入力チャンネル数,出力チャンネル数,カーネルサイズ)
                conv1 = F.Convolution2D(3, 32, 5, pad=2),
                conv2 = F.Convolution2D(32, 32, 5, pad=2),
                conv3 = F.Convolution2D(32, 32, 5, pad=2),
                #Linear(入力次元,出力次元)
                fc1 = F.Linear(8192, 2048),
                fc2 = F.Linear(2048, output_dim)
        )

    def forward(self, x_data, y_data, train=True, gpu=-1):
        """NNのforward処理を行う関数"""
        #gpuにデータを送る
        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
        #入力データと教師データをVariable型に変換
        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=2, stride=2) 
        #dropoutをかましたうえでNNを形成
        h = F.dropout(F.relu(self.fc1(h)), train=train)
        y = self.fc2(h)
        #なにを返り値で返しているのか理解する
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


class CNN:
    """CNNsのモデルを学習させるためのクラス"""
    def __init__(self, data, target, output_dim, gpu=-1):
        #CNNのモデルをメンバに作成
        self.model = C3fc2(output_dim)
        #モデルの名前をメンバに作成
        self.model_name = 'cnn_model'
        #GPUを使える場合はGPUを使って学習させる
        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        """
        scikit-learnの関数
        split(データ, 教師データ, test_size=0.1なら10%のデータを検証用にする)
        """
        self.x_train, \
        self.x_test, \
        self.y_train, \
        self.y_test = train_test_split(data, target, test_size=0.1)

        #学習用とテスト用のデータ数を保存
        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        #最適化手法を選択、パラメータの値を初期化
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def predict(self, x_data, gpu=-1):
        #このmodelの関数がなにをしているのか確認する
        return self.model.predict(x_data, gpu)

    def train_and_test(self, n_epoch=100, batchsize=100):
            epoch = 1
            best_accuracy = 0
            while epoch <= n_epoch:
                print('epoch', epoch)
                
                #なにをしているのか確認
                perm = np.random.permutation(self.n_train)
                sum_train_accuracy = 0
                sum_train_loss = 0
                for i in range(0, self.n_train, batchsize):
                    #伝播するデータをデータセットから分けて代入
                    x_batch = self.x_train[perm[i:i+batchsize]]
                    y_batch = self.y_train[perm[i:i+batchsize]]
                    #何をしているのか理解する
                    real_batchsize = len(x_batch)
                    #最適化手法の勾配を初期化する
                    self.optimizer.zero_grads()
                    #モデルにデータを伝播
                    loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                    loss.backward()
                    self.optimizer.update()
                    #何をしているのか理解する
                    sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                    sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

                print('train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train))
                #evaluation
                sum_test_accuracy = 0
                sum_test_loss = 0
                for i in range(0, self.n_test, batchsize):
                    x_batch = self.x_test[i : i+batchsize]
                    y_batch = self.y_test[i : i+batchsize]

                    real_batchsize = len(x_batch)

                    loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                    sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                    sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

                print('test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test))
                epoch += 1

    def dump_model(self):
        #モデルのパラメータを保存する
        self.model.to_cpu()
        #pickleの使い方を確認
        pickle.dump(self.model, open(self.model_name, 'wb'), -1)

    def load_model(self):
        #モデルのパラメータを読み込む
        self.model = pickle.load(open(self.model_name, 'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        #何をしているのか確認
        self.optimizer.setup(self.model.collect_parameters())



