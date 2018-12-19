
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:31:04 2017

@author: zhouhang
"""

import tensorflow as tf  
import numpy as np  
  
x = tf.placeholder(tf.float32, shape=[None, 1])  
y = 4 * x + 4  
  
w = tf.Variable(tf.random_normal([1], -1, 1))  
b = tf.Variable(tf.zeros([1]))  
y_predict = w * x + b  
  
  
loss = tf.reduce_mean(tf.square(y - y_predict))  
optimizer = tf.train.GradientDescentOptimizer(0.5)  
train = optimizer.minimize(loss)  
  

train_steps = 100  
checkpoint_steps = 50   
  
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b  
x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))  
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    for i in range(train_steps):  
        sess.run(train, feed_dict={x: x_data})  
        if (i + 1) % checkpoint_steps == 0:  
            savepaty=saver.save(sess,'model.ckpt', global_step=i+1) 
            print(savepaty)
