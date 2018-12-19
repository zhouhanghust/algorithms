# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import *
from sklearn import preprocessing

X_train,X_test,Y_train,Y_test = load_data()
sc = preprocessing.StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

learning_rate = tf.Variable(0.001,dtype=tf.float32)
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {
    'out':tf.Variable(tf.truncated_normal([n_hidden,n_classes],dtype=tf.float32))
}
biases = {
    'out':tf.Variable(tf.truncated_normal([n_classes]))
}

def RNN(x,weights,biases):
    # x = tf.unstack(x,n_steps,1)
    # lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    lstm_cell = rnn.core_rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
    outputs = tf.reshape(outputs, [-1,n_steps,n_hidden])
    return tf.matmul(outputs[:,-1,:],weights['out']+biases['out'])
    # return tf.matmul(final_state[1],weights['out']+biases['out'])

pred = RNN(x,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size <= len(X_train):
        # sess.run(learning_rate.assign(0.001*(0.95**(step-1))))
        batch_x,batch_y = X_train[(step-1) * batch_size:step * batch_size,],Y_train[(step-1) * batch_size:step * batch_size]
        batch_x = batch_x.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter"+str(step*batch_size)+",Minibatch Loss = "+ \
                  "{0:.6f}".format(loss)+",Training Accuracy= "+ \
                  "{0:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")
    X_test = X_test.reshape((-1,n_steps,n_input))
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))












