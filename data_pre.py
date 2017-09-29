#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/10 17:41
# @Author  : Zoe
# @Site    : 
# @File    : data_pre.py
# @Software: PyCharm Community Edition
import numpy as np
import pandas as pd

class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 reshape=False):
        self._num_examples = images.shape[0]
        self._images = images
#TODO: one-hot encode
        df=pd.get_dummies(labels)
        labels=(np.array(df)).astype('float')
#              
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def input_data():
    import h5py
    import pandas as pd

    h5f = h5py.File('F:\\PycharmProjects\\github\\CNN_series\\GAN\\Datas\\train.h5', 'r')
    data = h5f['train_data'][:]
    temp = np.zeros([data.shape[0], 24 * 24 - data.shape[1]])
    data = np.hstack((data, temp))
    label = h5f['train_label'][:]
   
    label=pd.get_dummies(label)
    label = np.array(label)


    data = DataSet(np.array(data), np.array(label.reshape([-1, 2])))
    return data