# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

X = tf.random_normal(shape=[10, 5, 6], dtype=tf.float32)
# X = tf.reshape(X, [-1, 5, 6])
weights={
         'in':tf.Variable(tf.random_normal([6,20])),
         'out':tf.Variable(tf.random_normal([20,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[20,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
input = tf.reshape(X,[-1,6])
input_rnn = tf.matmul(input,weights['in'])+biases['in']
input_rnn = tf.reshape(input_rnn,[-1,5,20])

cell = tf.nn.rnn_cell.BasicLSTMCell(20,forget_bias=1.0,state_is_tuple=True)  # 也可以换成别的，比如GRUCell，BasicRNNCell等等
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=1.0)
lstm_multi = tf.nn.rnn_cell.MultiRNNCell([cell] * 3, state_is_tuple=True)
state = lstm_multi.zero_state(10, tf.float32)
# state = state.zero_state(3,tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(lstm_multi, input_rnn, initial_state=state, time_major=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(final_state)))
    # print(sess.run(output[:,-1,:])==sess.run(final_state[1]))
    # print(sess.run(final_state[1]))
    print(sess.run(tf.shape(outputs)))
    print(sess.run(outputs[0,-1,:])==sess.run(final_state[-1][1][-1][0]))