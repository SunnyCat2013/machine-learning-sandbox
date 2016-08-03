# -*- coding: utf-8 -*-
# Author: IceBear
# Email: lizhenyang_2008@163.com
# Description: CHANGING WHEN IT IS EDITED
# Date: 20160803 10:22:06

import tensorflow as tf


# variables that can be changed here
IMAGE_HEIGHT = 20
IMAGE_WIDTH = IMAGE_HEIGHT
IMAGE_PIXEL = IMAGE_HEIGHT * IMAGE_WIDTH

CLASSES = 62





# main function 
def main():
    # 1. input layer
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXEL])
    y_ = tf.placeholder(tf.float32, [None, CLASSES])

    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 2. convolution layer 1
    W_conv1 = weight_variable([3, 3, 1, 128], 'W_conv1')
    b_conv1 = bias_variable([128], 'b_conv1') # 这里的128个输出层中，同层的输出都共享同一个偏置吗？
    h_conv1 = tf.relu(conv2d(x_image, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    # 3. convolution layer 2
    W_conv2 = weight_variable([3, 3, 128, 128], 'W_conv2')
    b_conv2 = bias_variable([128], 'b_conv2')
    h_conv2 = tf.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # 4. pool layer 1
    h_pool1 = max_pool_2x2(h_conv2)

    # 5. convolution layer 3
    W_conv3 = weight_variable([3, 3, 128, 256], 'W_conv3')
    b_conv3 = bias_variable([256],'b_conv3')
    h_conv3 = tf.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # 6. convolution layer 4
    W_conv4 = weight_variable([3, 3, 256, 256], 'W_conv4')
    b_conv4 = bias_variable([256], 'b_conv4')
    h_conv4 = tf.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # 7. pool layer 2
    h_pool2 = max_pool_2x2(h_conv4)

    # 8. convolution layer 5
    W_conv5 = weight_variable([3, 3, 256, 512], 'W_conv5')
    b_conv5 = bias_variable([512], 'b_conv5')
    h_conv5 = tf.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # 9. convolution layer 6
    W_conv6 = weight_variable([3, 3, 512, 512], 'W_conv6')
    b_conv6 = bias_variable([512], 'b_conv6')
    h_conv6 = tf.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # 10.convolution layer 7
    W_conv7 = weight_variable([3, 3, 512, 512], 'W_conv7')
    b_conv7 = bias_variable([512], 'b_conv7')
    h_conv7 = tf.relu(conv2d(h_conv6, W_conv7) + b_conv7)

    # 11.pool layer 3
    h_pool3 = max_pool_2x2(h_conv7)

    # 12.full connected layer 1
    W_fc1 = weight_variable([3 * 3 * 512, 2048], 'W_fc1')
    b_fc1 = bias_variable([2048], 'b_fc1')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 3 *  3 * 512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # 13.dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer (full connected layer 2)
    W_fc2 = weight_variable([2048, CLASSES], 'W_fc2')
    b_fc2 = bias_variable([CLASSES], 'b_fc2')
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# weight
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# bias
def bias_variable(shape, name):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


if __name__ == '__main__':
    main()
