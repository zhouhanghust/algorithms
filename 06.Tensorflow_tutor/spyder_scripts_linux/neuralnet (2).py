# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

train_samples = 200
test_samples = 60

def model(X,hidden_weights1,hidden_bias1,ow):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X,hidden_weights1)+hidden_bias1)
    return tf.matmul(hidden_layer,ow)

dsX = np.expand_dims(np.linspace(-1,1,train_samples+test_samples),1)
dsY = 0.4*np.power(dsX,2) + 2*dsX + np.random.randn(*dsX.shape)*0.22 + 0.8

# plt.figure()
# plt.title('Original data')
# plt.scatter(dsX,dsY)
# plt.show()

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

hw1 = tf.Variable(tf.truncated_normal([1,10],stddev=0.1))
ow = tf.Variable(tf.truncated_normal([10,1],stddev=0.1))

b = tf.Variable(tf.truncated_normal([10],stddev=0.1))
model_y = model(X,hw1,b,ow)

cost = tf.reduce_mean(tf.pow(model_y-Y,2))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    costs = []
    for epoch in range(300):
        sess.run(train_op,feed_dict={X:dsX,Y:dsY})
        costs.append(sess.run(cost,feed_dict={X:dsX,Y:dsY}))

    plt.figure()
    plt.title("Trainingloss lower and lower!")
    plt.plot(costs)
    plt.show()
