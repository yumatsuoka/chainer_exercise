#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画像と教師データを含むcsvを読み込み、４次元のテンソル型のリストを作成する。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from PIL import Image
import numpy as np
import six.moves.cPickle as pickle

class Input_data:
    def __init__(self, data_list):
        self.data_list = data_list
        self.data = None
        self.target = None
        self.pit = 123#データセットシャッフルするときの乱数のシード
        self.create_tensor()
        
    def create_tensor(self):
        """csvから画像と対応する教師データを読み込む """
        #csvの中身を全部読み込む。
        csv_data_raw = pd.read_csv(self.data_list, header=None)
        #読み込んだデータセットを列でシャッフル
        np.random.seed(self.pit)
        csv_data = csv_data_raw.reindex(np.random.permutation(csv_data_raw.index))
        #教師データを取得 
        self.data = np.array([np.asarray(Image.open(csv_data[0][i], 'r')) for i in range(len(csv_data))])
        self.target = np.array([csv_data[1][i] for i in range(len(csv_data))])
        #画素値を255で除算
        #self.data = np.transpo((self.data.astype(np.float32) / 255.0), (1,0,2))
        self.data = self.data.astype(np.float32) / 255.0
        self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.target = self.target.astype(np.int32)
    
    def increase_image(self):
        pass
    def gamma_correction(self):
        pass
    def smoothing_filter(self):
        pass
    def contrast_adjustment(self):
        pass    
    def add_noise(self):
        pass
    def flip_horizontal(self):
        pass
    def dump_dataset(self):
        pickle.dump((self.data, self.target, self.index2name), open(self.dump_name, 'wb'), -1)
    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, 'rb'))

if __name__ == '__main__':
    data_list = "/home/yuma/programing/github/ikemen_check/target/bijo_target.csv"
    input_data = Input_data(data_list)
    
