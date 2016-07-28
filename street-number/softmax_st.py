# -*- coding: utf-8 -*-
# 我打算使用一个简单的softmax回归模型来对街道的号码的分类工作做一个测试


from PIL import Image
import numpy as np
import tensorflow as tf

import zipfile


# initialization

NUM_CLASSES = 62
IMAGE_SIZE = 20
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def main():
    # active a inter session
    sess = tf.InteractiveSession()

    # x is the place holder for the input samples, y_ is the ground truth.
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # Variables
    W = tf.Variable(tf.zeros[IMAGE_PIXELS, NUM_CLASSES])
    b = tf.Variable(tf.zeros[NUM_CLASSES])

    # initial the Variables
    sess.run(tf.initialize_all_variables())

    # predict function
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # loss function by cross-entropy error
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # optimizer and learning rate
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # import street images which containt numbers
    stnum = read_st_img('street-data/trainResized.zip')

    # close session
    sess.close()

def read_st_img(sfile):
    # read zip file
    archive = zipfile.ZipFile(sfile, 'r')
    file_list = archive.namelist()

    imgs = list()

    del file_list[0] # delete directory name
    for f in file_list:
        imgfile = archive.open(f, 'r')
        img = Image.open(imgfile)
        imgfile.close()

        imgarr = np.asarray(img.convert('L'))
        img.close()

        img = imgarr.ravel()
        imgs.append(img)

    return np.asarray(imgs)



if __name__ == '__main__':
    main()
