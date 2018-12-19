#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 15:46:35 2017

@author: zhouhang
"""

import tensorflow as tf
import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

N = 210
K = 2

MAX_ITERS = 1000
cut = int(N*0.7)

start = time.time()

data,y_label = make_circles(n_samples=N,shuffle=True,noise=0.12,factor=0.4)
tr_data,tr_label = data[:cut],y_label[:cut]
te_data,te_label = data[cut:],y_label[cut:]

fig,ax = plt.subplots()
ax.scatter(tr_data.transpose()[0],tr_data.transpose()[1],marker='o',s=100,c=tr_label,cmap=plt.cm.coolwarm)
plt.show()

points = tf.constant(data)
cluster_assignments = tf.Variable(tf.zeros([N],dtype=tf.int64))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    test = []
    for i,j in zip(te_data,te_label):
        distances = tf.reduce_sum(tf.square(tf.subtract(i,tr_data)),axis=1)
        neighbor = tf.arg_min(distances,0)
        test.append(tr_label[sess.run(neighbor)])
print(test)
fig,ax = plt.subplots()
ax.scatter(te_data.transpose()[0],te_data.transpose()[1],marker='o',s=100,c=test,cmap=plt.cm.coolwarm)
plt.show()
    
end = time.time()
print("Found in %.2f seconds"%(end-start))
print("Cluster assignments:",test)











