# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

N = 300
K = 2
centers = [(-4,-4),(4.5,3.5)]

data,y_label = make_blobs(n_samples=N,centers=centers
                            ,n_features=2,cluster_std=0.8,shuffle=True,random_state=42)


lr = tf.Variable(0.001,dtype=tf.float32)
X = tf.placeholder(shape=[None,2],dtype=tf.float32)
Y = tf.placeholder(shape=[None],dtype=tf.float32)

W = tf.Variable(tf.truncated_normal([2,1],stddev=0.1))
b = tf.Variable(tf.zeros([1]))

Z = tf.matmul(X,W) + b
a = tf.exp(-Z)
y_out = tf.squeeze(1.0 / (1.0 + a))

y_pred = tf.cast((y_out > 0.5),tf.int32)

cost = tf.reduce_mean(-(Y*tf.log(y_out) + (1-Y)*tf.log(1-y_out)))
optimizer =  tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # costs = []
    sess.run(init)
    for epoch in range(150):
        sess.run(lr.assign(0.001 * (0.95)**epoch))
        sess.run(train_op,feed_dict={X:data,Y:y_label})
        # costs.append(sess.run(cost,feed_dict={X:data,Y:y_label}))
    pred_label = sess.run(y_pred,feed_dict={X:data})

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_label))])

    x1_min,x1_max = data[:,0].min()-1,data[:,0].max()+1
    x2_min,x2_max = data[:,1].min()-1,data[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.005),
                          np.arange(x2_min,x2_max,0.005))
    Z_value = sess.run(y_pred,feed_dict={X:np.array([xx1.ravel(),xx2.ravel()]).T})
    Z_value = Z_value.reshape(np.shape(xx1))
    plt.contourf(xx1,xx2,Z_value,alpha = 0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.scatter(np.asarray(data).transpose()[0]
                ,np.asarray(data).transpose()[1],marker='o',s=50,c=y_label)

    plt.show()

    # plt.plot(costs)
    # plt.show()




