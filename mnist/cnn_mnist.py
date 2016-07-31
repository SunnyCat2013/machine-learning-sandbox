# -*- coding: utf-8 -*-
# Author: IceBear
# Email: lizhenyang_2008@163.com
# Description: Build a multilayer convolution network for mnist
# Date: 20160731 11:36:01

import tensorflow as tf


# weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


## reshape x
x_image = tf.reshape(x, [-1, 28, 28, 1])


## convolution and pooling
h_conv1 = tf.nn.relu(conv2d(x_imge, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

## convolution and pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
