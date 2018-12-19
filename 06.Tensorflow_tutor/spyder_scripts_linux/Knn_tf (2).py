#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:31:59 2017

@author: zhouhang
"""

import tensorflow as tf
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

DATA_TYPE = 'blobs'

MAX_ITERS = 1000
start = time.time()

N = 300
K = 4
centers = [(-2,-2),(-2,1.5),(1.5,-2),(2,1.5)]

data,y_label = make_blobs(n_samples=N,centers=centers
                            ,n_features=2,cluster_std=0.8,shuffle=True,random_state=42)

#fig,ax = plt.subplots(2,2,sharex=True,sharey=True)
#ax[0,0].scatter(np.asarray(centers).transpose()[0],np.asarray(centers).transpose()[1],marker='o',s=200)
#ax[1,1].scatter(np.asarray(data).transpose()[0],np.asarray(data).transpose()[1],marker='o',s=200,c=y_label)
#plt.show()

    
points = tf.constant(data,dtype=tf.float32)
cluster_assignments = tf.Variable(tf.zeros([N],dtype=tf.int64))

centroids = tf.Variable(tf.slice(points,[0,0],[K,2]))


rep_centroids = tf.reshape(tf.tile(centroids,[N,1]),[N,K,2])
rep_points = tf.reshape(tf.tile(points,[1,K]),[N,K,2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),axis=2)
best_centroids = tf.argmin(sum_squares,1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,cluster_assignments))    

def bucket_mean(data,bucket_ids,num_buckets):
    total = tf.unsorted_segment_sum(data,bucket_ids,num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data),bucket_ids,num_buckets)
    return total / count
    
means = bucket_mean(points,best_centroids,K)
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
                          centroids.assign(means),
                          cluster_assignments.assign(best_centroids))

init = tf.global_variables_initializer()    
with tf.Session() as sess:
    sess.run(init)

    iters = 0
    changed = True
    
    while changed and iters < MAX_ITERS:
        fig,ax = plt.subplots()
        iters += 1
        [changed,_] = sess.run([did_assignments_change,do_updates])
        [centers,assignments] = sess.run([centroids,cluster_assignments])
        ax.scatter(sess.run(points).transpose()[0],sess.run(points).transpose()[1]
                   ,marker='o',s=200,c=assignments,cmap=plt.cm.coolwarm)
        ax.scatter(centers[:,0],centers[:,1],marker='^',s=550,c=np.arange(K),
                   cmap=plt.cm.coolwarm)
        ax.set_title('Iteration '+str(iters))
        plt.show()
#        plt.savefig("kmeans"+str(iters)+".png")
    
#    fig,ax = plt.subplots()
#    ax.scatter(sess.run(points).transpose()[0],sess.run(points).transpose()[1],
#               marker='o',s=200,c=assignments,cmap=plt.cm.coolwarm)
#    ax.scatter()
#    plt.show()
end = time.time()
print("Found in %.2f seconds"%(end-start),iters,"iterations")
print("Centroids:")
print(centers)
print("Cluster assignments:",assignments)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    
    
    
    
    
    
    
    