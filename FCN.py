#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 8/8/17 PM10:37
# @Author : Zoe
# @Site :
# @File : CNN_series.py
# @Software: PyCharm Community Edition

import tensorflow as tf


NUM_CHANNELS=1
# NUM_LABELS=10

CONV1_DEEP=16
CONV1_SIZE_row=1
CONV1_SIZE_col=2

CONV2_DEEP=16
CONV2_SIZE_row=1
CONV2_SIZE_col=2

CONV3_DEEP=16
CONV3_SIZE_row=1
CONV3_SIZE_col=2

# FC_SIZE=1024

OUTPUT_NODE=1

def model(input_tensor,phase,IMAGE2):
    
#    print(input_tensor.get_shape())
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row, CONV1_SIZE_col, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
#        print(conv1.get_shape())
        bn_fc1 = tf.layers.batch_normalization(conv1, training=phase)
        relu1 = tf.nn.relu(tf.nn.bias_add(bn_fc1,conv1_biases))

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row, CONV2_SIZE_col, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")

        bn_fc2 = tf.layers.batch_normalization(conv2, training=phase)
        relu2 = tf.nn.relu(tf.nn.bias_add(bn_fc2,conv2_biases))

    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row, CONV3_SIZE_col, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")

        bn_fc3 = tf.layers.batch_normalization(conv3, training=phase)
        relu3 = tf.nn.relu(tf.nn.bias_add(bn_fc3,conv3_biases))

    with tf.variable_scope('layer4-pool'):
        fc_weights = tf.get_variable("weights", [CONV3_DEEP, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        globalpool = tf.nn.avg_pool(relu3, ksize=[1, 1, relu3.get_shape()[2], 1], strides=[1, 1, relu3.get_shape()[2], 1], padding="VALID")
        reshaped = tf.reshape(globalpool, [-1, CONV3_DEEP])
        fc=tf.matmul(reshaped, fc_weights) + fc_biases
        logit = tf.nn.sigmoid(tf.matmul(reshaped, fc_weights) + fc_biases)

    return fc,logit



