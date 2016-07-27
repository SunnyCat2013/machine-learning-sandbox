# -*- coding:utf-8 -*-

#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()

# shape=[None, 784]，是因为样本个数不知道，但是知道每个样本的维度是784.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# initial variables to the specified values, in this case they are all set to zeros.
sess.run(tf.initialize_all_variables())

#Predicted Class and Cost Function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train
for i in range(10000):
    if i % 100 == 0:
        print 'training...', i
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_: batch[1]})

# evaluation
## evaluation componant
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print accuracy
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# close the Session
sess.close()
