# -*- coding:utf-8 -*-

import tensorflow as tf

weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32]))
    'wc2':tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
    'wd1':tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
    'out':tf.Variable(tf.random_normal([1024,n_class]))
}
bias = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal[64]),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.truncated_normal([n_classes]))
}

XXX = tf.Variable(bias['bc1'].initialized_value())

conv_layer_1 = conv2d(insize,weights['wc1'],bias['bc1'])

