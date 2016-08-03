# -*- coding: utf-8 -*-
# 我打算使用一个简单的softmax回归模型来对街道的号码的分类工作做一个测试


from PIL import Image
from io import BytesIO
import numpy as np
import os
import sys
import tensorflow as tf

import zipfile


# initialization

NUM_CLASSES = 62
IMAGE_SIZE = 20
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

EPOCH = 100000
BATCH = 1000
MODEL_FILE = "./Softmax_%d_epoch_%d_batch.ckpt" % (EPOCH, BATCH)

def main(model_file):
    # active a inter session
    sess = tf.InteractiveSession()

    # x is the place holder for the input samples, y_ is the ground truth.
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    # Variables
    W = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]), name="W")
    b = tf.Variable(tf.zeros([NUM_CLASSES]), name="b")

    # initial the Variables
    sess.run(tf.initialize_all_variables())
    
    # saver
    saver = tf.train.Saver()

    # predict function
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # if there are a valid model file
    if os.path.exists(model_file):
        print 'restore model file:', model_file
        saver.restore(sess, model_file)
    else:
        # loss function by cross-entropy error
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # optimizer and learning rate
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # import street images which containt numbers
        stnum_imgs = read_st_img('street-data/trainResized.zip')
        stnum_labs = read_st_label('street-data/trainLabels.csv')

        stnum_obj = simpleDataSet(stnum_imgs, stnum_labs)


        # train!!
        for i in range(EPOCH):
            if i % 100 == 0:
                print 'training...', i
            batch = stnum_obj.next_batch(BATCH)
            train_step.run(feed_dict={x:batch[0], y_:batch[1]})

        # save variables
        save_path = saver.save(sess, MODEL_FILE)
        print 'Model saved as', save_path


    # prediction
    stnum_test = read_st_img('street-data/testResized.zip', testFile=True)
    classification = sess.run(tf.argmax(y, 1), {x: stnum_test})

    # save result
    with open('result_softmax_%d_epoch_%d_batch.csv'%(EPOCH, BATCH), 'w') as of:
        print >>of, 'ID,Class'
        keys = read_st_label('street-data/trainLabels.csv', get_key_ind=True)
        i = 6284
        for c in classification:
            print >>of, '%d,%s'%(i, keys[c])
            i += 1


    # close session
    sess.close()

def read_st_img(sfile, testFile=False):
    # read zip file
    archive = zipfile.ZipFile(sfile, 'r')
    file_list = archive.namelist()

    imgs = list()

    del file_list[0] # delete directory name
    if testFile:
        file_list.sort(key=lambda x:int(x[12:-4]))
    else:
        file_list.sort(key=lambda x:int(x[13:-4]))
    with open(sfile.split('/')[-1] + '_detail.txt', 'w') as of:
        for f in file_list:
            print >>of, f

    outf = open(sfile + '_reading_order.txt', 'w')
    for f in file_list:
        print >>outf, f
        try:
            imgfile = archive.open(f, 'r')
            img = Image.open(imgfile)
            imgfile.close()

            imgarr = np.asarray(img.convert('L'))
            img.close()

            img = imgarr.ravel()
            imgs.append(img)
        except:
            imgfile = BytesIO(archive.read(f, 'r'))
            img = Image.open(imgfile)
            #imgfile.close()

            imgarr = np.asarray(img.convert('L'))
            #img.close()

            img = imgarr.ravel()
            imgs.append(img)
    outf.close()

    temp = np.asarray(imgs, dtype=np.float32)

    return np.multiply(temp, 1.0 / 255.0)


def read_st_label(sfile, one_hot=True, get_key_ind=False):
    # read labels
    #sfile = 'street-data/trainLables.csv'
    # get the labels' indices and the labels
    # return one_hot format
    dlist = dict()
    with open(sfile, 'r') as inf:
        lines = inf.readlines()
        del lines[0]
        ln = len(lines)
        labels = list()
        for l in lines:
            l = l.strip().split(',')
            key = l[1]
            labels.append(key)
            dlist[key] = 1
    keys = dlist.keys()
    keys.sort()
    if get_key_ind:
        return keys # return the labels array list, in which keys[i] = labels
    dkey_index = dict()
    for i, k in enumerate(keys):
        dkey_index[k] = i


    '''
    for k in keys:
        print k, dkey_index[k]
    print ''
    '''
    num_labels = len(labels)
    num_classes = len(keys)

    #print num_labels, num_classes
    numer_labels = np.zeros(num_labels, dtype=np.uint8)
    for i, l in enumerate(labels):
        numer_labels[i] = dkey_index[l]

    index_offset = np.arange(num_labels) * num_classes

    one_hot = np.zeros((num_labels, num_classes), dtype=np.uint8)

    one_hot.flat[index_offset + numer_labels] = 1

    '''
    # test
    for i in range(62):
        print i,
    print ''
    for i in range(10):
        print i
        print labels[i]
        print dkey_index[labels[i]]
        for l in one_hot[i]:
            print l,
        print ''
    '''

    return one_hot


class simpleDataSet(object):
    def __init__(self, images, labels):
        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

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
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


if __name__ == '__main__':
    model_file = ""
    if len(sys.argv) > 1: # 如果已经有训练好的模型
        print sys.argv[1]
        model_file = sys.argv[1]
    main(model_file)
    #test = read_st_img('street-data/trainResized.zip')
    #print test[0][0]
    #read_st_label('street-data/trainLabels.csv')

