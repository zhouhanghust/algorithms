
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:45:14 2017

@author: zhouhang
"""

import tensorflow as tf  
import numpy as np
import os
  
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
    # sess.run(tf.global_variables_initializer()) # 不一定要加，因为后面的restore包含了init
    ckpt = tf.train.get_checkpoint_state('')
    ckpt2 = tf.train.get_checkpoint_state(os.path.dirname(__file__))
   # saver.restore(sess, ckpt.model_checkpoint_path)
    print(ckpt.model_checkpoint_path)
    print(ckpt2.model_checkpoint_path)
#     saver.restore(sess,"./model.ckpt-100")
    saver.restore(sess,ckpt2.model_checkpoint_path)
    print(sess.run(w))
    print(sess.run(b))







