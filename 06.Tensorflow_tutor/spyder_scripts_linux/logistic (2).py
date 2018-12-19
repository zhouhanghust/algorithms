# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv("CHD.csv",header=0)
# print(df.head(10))

# lr = tf.Variable(initial_value=0.001,dtype=tf.float32)
training_epochs = 5
batch_size = 100
display_step = 1
b = np.zeros((100,2))


# x = tf.placeholder(dtype=tf.float32,shape=[None,1])
# y = tf.placeholder(dtype=tf.float32,shape=[None,2])

# w = tf.Variable(tf.zeros([1,2]))
# b = tf.Variable(tf.zeros([2]))
#
# activation = tf.nn.softmax(tf.matmul(x,w)+b)
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation),axis=1))
#
# optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
#
# init = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init)
    # for epoch in range(training_epochs):
        # sess.run(lr.assign(0.001*(0.95**epoch))
        # total_batch = 400//batch_size
        # for i in range(total_batch):
            temp = tf.one_hot(indices = df['chd'].values,depth=10
                              ,on_value=1,off_value=0,axis=1,name='a')
            print(sess.run(temp))



