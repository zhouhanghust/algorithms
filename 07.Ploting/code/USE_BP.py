# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: USE_BP.py
@time: 2018/2/14 15:37
@Software: PyCharm Community Edition
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error


maxmin = pd.read_csv('maxmin.csv',index_col=0)
min_ZDJ = maxmin.iloc[1,0]
max_ZDJ = maxmin.iloc[0,0]

data = sio.loadmat('ShangZZS.mat')
X_train = data['X_train']
X_train = X_train[::-1,:]
X_test = data['X_test']
X_test = X_test[::-1,:]
y_train = data['y_train'][:,0]
y_train = y_train[::-1]
y_test = data['y_test'][:,0]
y_test = y_test[::-1]

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]
#############################################################################
#define graph
input_size = X_train.shape[1]
hidden_size = 6


X = tf.placeholder(dtype=tf.float32,shape=[None,input_size])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])

lr = tf.Variable(0.1,dtype=tf.float32)

W1 = tf.Variable(initial_value=tf.truncated_normal([input_size,hidden_size]))
b1 = tf.Variable(initial_value=tf.zeros([1,hidden_size])+0.1)

W2 = tf.Variable(initial_value=tf.truncated_normal([hidden_size,1]))
b2 = tf.Variable(initial_value=tf.zeros([1,1])+0.1)


l1 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)
output = tf.nn.sigmoid(tf.matmul(l1,W2)+b2)


def train():
    loss = tf.reduce_mean(tf.square(y-output))
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        max_iter = 500
        for epoch in range(max_iter):
            sess.run(lr.assign(0.1*(0.95**(epoch))))
            _,_loss = sess.run([train_op,loss],feed_dict={X:X_train,y:y_train})

            print('Epoch {0} loss {1:.3f}'.format(epoch+1,_loss))
        print('model_save',saver.save(sess,'model_save/model.ckpt',global_step=max_iter))
        print("The train has finished!")
# train()


def pred():
    global y_train,y_test
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'model_save/model.ckpt-500')
        prediction_train = sess.run(output,feed_dict={X:X_train})
        prediction_test = sess.run(output,feed_dict={X:X_test})

    y_train = y_train * (max_ZDJ - min_ZDJ) + min_ZDJ
    y_test = y_test * (max_ZDJ - min_ZDJ) + min_ZDJ

    prediction_train = prediction_train * (max_ZDJ - min_ZDJ) + min_ZDJ
    prediction_test = prediction_test * (max_ZDJ - min_ZDJ) + min_ZDJ

    print("Mean_Squared_Error :{0:.3f}".format(mean_squared_error(y_test, prediction_test)))

    mse = mean_squared_error(y_test, prediction_test)
    rmse = np.sqrt(mse)
    print('RMSE:', rmse)
    mape = np.abs((y_test - prediction_test) / y_test)
    mape = np.mean(mape)
    print('MAPE:', mape)







    plt.figure()
    font = FontProperties(fname="simhei.ttf", size=16)
    plt.plot(np.arange(len(y_train) + len(y_test)), np.vstack([y_train, y_test]), c='k',
             label='真实值', linewidth=2.0)
    plt.plot(np.arange(len(y_train) + len(y_test)), np.vstack([prediction_train, prediction_test]), c='b',
             label='BP预测值', linestyle='-', linewidth=2.0, marker='o', markersize=10)
    plt.xlabel('日期（2017/12/12-2018/02/09）', fontproperties=font)
    plt.ylabel('上证指数最低价', fontproperties=font)
    plt.legend(prop=font)
    plt.show()
    # sio.savemat('BP.mat',{'pred':np.vstack([prediction_train, prediction_test])})
pred()




