# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

df = pd.read_csv("data/boston.csv",header=0)

# print(df.head(10))

# f,ax1 = plt.subplots()
#
# y = df['MEDV']
#
# for i in range(1,8):
#     number = 420 + i
#     ax1.locator_params(nbins=3)
#     ax1=plt.subplot(number)
#     plt.title(list(df)[i])
#     ax1.scatter(df[df.columns[i]],y)
# plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
# plt.show()

X = tf.placeholder(shape=[None,2],dtype=tf.float32,name='X')
Y = tf.placeholder(shape=[None],dtype= tf.float32,name='Y')

w = tf.Variable(tf.random_normal([2,1],stddev=0.01),dtype=tf.float32,name='b0')
b = tf.Variable(tf.random_normal([1],stddev=0.01),dtype=tf.float32,name='b1')

y_reg = tf.squeeze(tf.matmul(X,w)+b)

cost = tf.reduce_mean(tf.pow(Y-y_reg,2))

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    xvalues = df[[df.columns[2],df.columns[4]]].values.astype(np.float32)
    yvalues = df[df.columns[12]].values.astype(np.float32)
    costm = []
    for epoch in range(1,5000):
        sess.run(train_op,feed_dict={X:xvalues,Y:yvalues})
        costm.append(sess.run(cost,feed_dict={X:xvalues,Y:yvalues}))

plt.plot(costm)
plt.show()



