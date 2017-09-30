#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/10 13:32
# @Author  : Zoe
# @Site    :
# @File    : MLP.py
# @Software: PyCharm Community Edition
import tensorflow as tf

NUM_CHANNELS = 1
# INPUT_NODE = 1222
OUTPUT_NODE = 1
HIDDEN1 = 32
HIDDEN2 = 32
HIDDEN3 = 64


def model(input_data, keep_prob, INPUT_NODE):
    input_data=tf.reshape(input_data,shape=[-1,INPUT_NODE])
    with tf.variable_scope("hidden-layer1"):
        weight1 = tf.get_variable("weight", [INPUT_NODE, HIDDEN1],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases1 = tf.get_variable("bias", [HIDDEN1],
                                  initializer=tf.constant_initializer(0.1))
        hidden1 = tf.nn.relu(tf.matmul(input_data, weight1) + biases1)

        hidden1_drop = tf.nn.dropout(hidden1, 0.5)
    with tf.variable_scope("hidden-layer2"):
        weight2 = tf.get_variable("weight", [HIDDEN1, HIDDEN2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases2 = tf.get_variable("bias", [HIDDEN2],
                                  initializer=tf.constant_initializer(0.1))
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, weight2) + biases2)

        hidden2_drop = tf.nn.dropout(hidden2, 0.5)
    with tf.variable_scope("hidden-layer3"):
        weight3 = tf.get_variable("weight", [HIDDEN2, HIDDEN3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases3 = tf.get_variable("bias", [HIDDEN3],
                                  initializer=tf.constant_initializer(0.1))
        hidden3 = tf.nn.relu(tf.matmul(hidden2_drop, weight3) + biases3)

        hidden3_drop = tf.nn.dropout(hidden3, 0.5)
    with tf.variable_scope("output-layer"):
        weight4 = tf.get_variable("weight", [HIDDEN3, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases4 = tf.get_variable("bias", [OUTPUT_NODE],
                                  initializer=tf.constant_initializer(0.1))
        fc=tf.matmul(hidden3_drop, weight4) + biases4
        logist = tf.nn.sigmoid(tf.matmul(hidden3_drop, weight4) + biases4)
    return fc,logist
