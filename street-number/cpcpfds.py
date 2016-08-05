# -*- coding: utf-8 -*-
# Author: IceBear
# Email: lizhenyang_2008@163.com
# Description: CHANGING WHEN IT IS EDITED
# Date: 20160803 10:22:06

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from softmax_st import *

import os, sys

#from softmax_st import *

# variables that can be changed here
IMAGE_HEIGHT = 20
IMAGE_WIDTH = IMAGE_HEIGHT
IMAGE_PIXEL = IMAGE_HEIGHT * IMAGE_WIDTH

CLASSES = 62

EPOCH = 1
BATCH = 50

CONV_SIZE = 3

NET_TYPE = 'cpcpf'

MODEL_FILE = "./%s_CNN_%d_epoch_%d_batch.ckpt" % (NET_TYPE, EPOCH, BATCH)


# main function 
def main(model_file):
    # 1. input layer
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXEL])
    y_ = tf.placeholder(tf.float32, [None, CLASSES])

    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    POOL_SIZE = IMAGE_HEIGHT
    # 2. convolution layer 1
    W_conv1 = weight_variable([CONV_SIZE, CONV_SIZE, 1, 32], 'W_conv1')
    b_conv1 = bias_variable([32], 'b_conv1') # 这里的128个输出层中，同层的输出都共享同一个偏置吗？
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #h_pool1 = max_pool_2x2(h_conv1)

    # test
    h_pool1 = max_pool_2x2(h_conv1)
    POOL_SIZE = int((POOL_SIZE + 1) / 2)
    # 3. convolution layer 2
    W_conv2 = weight_variable([CONV_SIZE, CONV_SIZE, 32, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)
    POOL_SIZE = int((POOL_SIZE + 1) / 2)

    W_fc1 = weight_variable([POOL_SIZE * POOL_SIZE * 64, 1024], 'W_fc1')
    b_fc1 = bias_variable([1024], 'b_fc1')

    h_pool3_flat = tf.reshape(h_pool2, [-1, POOL_SIZE * POOL_SIZE * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob1 = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

   # 16.out layer(full connected layer 3)
    W_fc3 = weight_variable([1024, CLASSES], 'W_fc3')
    b_fc3 = bias_variable([CLASSES], 'b_fc3')
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create session
    sess = tf.InteractiveSession()

    # initialize variables
    sess.run(tf.initialize_all_variables())

    # saver
    saver = tf.train.Saver()
    # if there are a valid model file
    if os.path.exists(model_file):
        ## restore variables
        print 'restore model from:', model_file
        saver.restore(sess, model_file)
    else:
        stnum_imgs = read_st_img('street-data/trainResized.zip')
        stnum_labs = read_st_label('street-data/trainLabels.csv')
        stnum_obj = simpleDataSet(stnum_imgs, stnum_labs)

        #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        for i in range(EPOCH):
            batch = stnum_obj.next_batch(BATCH)
            if i % 100 == 0:
                #print 'training...', 
                train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob1:1.0})
                print 'training step %d, the accuracy: %g'%(i, train_accuracy)
            train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob1:0.5})
        # save variables
        save_path = saver.save(sess, MODEL_FILE)
        print 'Model saved as:', save_path


    # test
    #test_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob1: 1.0})
    #print 'test accuracy: %g'%(test_accuracy)
    # prediction
    print 'reading data'
    stnum_test = read_st_img('street-data/testResized.zip', testFile=True)
    print 'predicting...'
    classification = sess.run(tf.argmax(y_conv, 1), {x: stnum_test, keep_prob1:1.0})

    # save result
    print 'saving...'
    with open('result_%s_cnn_%d_epoch_%d_batch.csv'%(NET_TYPE, EPOCH, BATCH), 'w') as of:
        print >>of, 'ID,Class'
        keys = read_st_label('street-data/trainLabels.csv', get_key_ind=True)
        i = 6284
        arr = "-\\|/"
        for c in classification:
            print arr[i % 4],
            print >>of, '%d,%s'%(i, keys[c])
            print '\r',
            i += 1

    sess.close()



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
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


if __name__ == '__main__':
    model_file = ''
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
        print 'using model', model_file
    main(model_file)
