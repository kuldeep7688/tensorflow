# -*- coding: utf-8 -*-

"""
Tutorial code for running softmax regression on MNIST using
Tensorflow.

By: Kuldeep Singh
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "/tmp/data"
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy_loss)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# running session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_x, batch_y = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(optimizer, feed_dict={x: batch_x, y_true: batch_y})

    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                            y_true: data.test.labels})

print "Accuracy: {:.4}%".format(ans*100)

