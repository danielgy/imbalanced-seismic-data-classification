# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:52:57 2018
@author: GY
"""
import tensorflow as tf


# Hyper-parameter
NUM_CHANNELS = 1
CONV1_DEEP = 32
CONV1_SIZE_row = 1
CONV1_SIZE_col = 3
CONV2_DEEP = 32
CONV2_SIZE_row = 1
CONV2_SIZE_col = 3
CONV3_DEEP = 64
CONV3_SIZE_row = 1
CONV3_SIZE_col = 3
FC_SIZE = 128
OUTPUT_NODE = 1


def model(input_tensor, phase):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE_row,
                                                   CONV1_SIZE_col,
                                                   NUM_CHANNELS,
                                                   CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP],
                                       initializer=tf.constant_initializer(0.1))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE_row,
                                                   CONV2_SIZE_col,
                                                   CONV1_DEEP,
                                                   CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool1 = tf.nn.max_pool(relu2, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope('layer5-conv3'):
        conv3_weights = tf.get_variable("weight", [CONV3_SIZE_row,
                                                   CONV3_SIZE_col,
                                                   CONV2_DEEP,
                                                   CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.1))

        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding="VALID")
        bn_conv3 = tf.layers.batch_normalization(tf.nn.bias_add(conv3, conv3_biases), training=phase)
        relu3 = tf.nn.relu(tf.nn.bias_add(bn_conv3, conv3_biases))

    with tf.variable_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    pool_shape = pool3.get_shape().as_list()
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool3, [-1, node])
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable("weights", [node, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable("bias", [FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        bn_fc1 = tf.layers.batch_normalization(tf.matmul(reshaped, fc1_weights)+fc1_biases, training=phase)
        fc1 = tf.nn.relu(bn_fc1)
        fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
        logit = tf.nn.sigmoid(tf.matmul(fc1, fc2_weights) + fc2_biases)

    return fc2, logit
