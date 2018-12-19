# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# df = pd.read_csv("szcfzs/sczs.csv",header=0)
# print(df['time'][1500:])
# data = df['spzs'][1500:]
# plt.figure()
# plt.plot(data)
# plt.show()

# data = np.array(df['spzs'])
# normalize_data=(data-np.mean(data))/np.std(data)
# normalize_data=normalize_data[:,np.newaxis]
# print(normalize_data.shape)

data = np.load('mc_data.npy')
data = data[:,np.newaxis]
sc = preprocessing.StandardScaler()
sc.fit(data)
data = sc.transform(data)
# print(data[:10])


time_step=30
rnn_unit=20
batch_size=60
input_size=1
output_size=1
lr=0.0006
xx,yy=[],[]
for i in range(len(data)-time_step):
    x=data[i:i+time_step]
    y=data[i+time_step]
    xx.append(x.tolist())
    yy.append(y.tolist())

# print(np.shape(xx[0]))

train_x = xx[:910]
train_y = yy[:910]
test_x = xx[910:]
test_y = yy[910:]
# print(np.shape(test_x[0]))
# print(len(train_x),len(train_y))
# print(train_x[1][-1],train_y[0])
# print(train_x[2][-1],train_y[1])
# print(train_x[-1][-1],train_y[-2])
# print(train_y[-1])
# print(len(test_y))

X = tf.placeholder(tf.float32, [None, time_step, input_size])
Y = tf.placeholder(tf.float32, [None , output_size])

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    # input_rnn = tf.reshape(X, [-1, time_step, input_size])  # new change
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,forget_bias=1.0,state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=1.0)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*1, state_is_tuple=True)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32,time_major=False)
    # output = tf.reshape(output_rnn, [-1,time_step,rnn_unit])

    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output_rnn[:,-1,:], w_out) + b_out

    return pred, final_states


def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(500):  # We can increase the number of iterations to gain better result.
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step % 10 == 0:
                    print("Number of iterations:", i, " loss:", loss_)
                    # print("model_save", saver.save(sess, 'model_save1\\modle.ckpt'))
                    # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    # if you run it in Linux,please use  'model_save1/modle.ckpt'
                step += 1
        print("model_save", saver.save(sess, 'model_save3\\modle3.ckpt'))
        print("The train has finished")


# train_lstm()


def pred_train():
    with tf.variable_scope("sec_lstm"):     #, reuse=True
        pred, _ = lstm(len(train_x))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save3\\modle3.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'

        predict = sess.run(pred, feed_dict={X: train_x})

        # prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(data))), data, color='b')
        plt.plot(list(range(len(train_x[0]), len(train_x[0])  + len(predict))), predict, color='r')
        plt.show()


# pred_train()

def pred_test():
    global test_y,train_y
    with tf.variable_scope("sec_lstm"):     #, reuse=True
        pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    prediction = []
    with tf.Session() as sess:
        saver.restore(sess, 'model_save3\\modle3.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        for epoch in range(len(test_y)):
            predict = sess.run(pred, feed_dict={X: np.array(test_x[epoch]).reshape(-1,30,1)})
            # print(predict.shape)
            prediction.append(np.squeeze(predict))

        test_y = np.squeeze(np.array(test_y))
        train_y = np.squeeze(np.array(train_y))
        # print(len(prediction),len(test_y))
        # print(np.shape(prediction),np.shape(test_y))
        # prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        # concaty = np.hstack([train_y[880:],test_y])
        # fig,ax = plt.subplots()
        # ax.plot(range(len(train_y[:880]),len(train_y[:880])+len(concaty)), concaty, color='b')
        # ax.plot(range(len(train_y),len(train_y)+len(prediction)), prediction, color='r')
        # plt.show()
        plt.figure()
        plt.plot(range(len(test_y)),test_y,color='b',label='test_y')
        plt.plot(range(len(prediction)),prediction,color='r',label='prediction')
        plt.xlabel("Last_60_data",size=15)
        plt.ylabel("normalized_data",size=15)
        plt.legend(loc="upper left")
        plt.show()
        pd.DataFrame(test_y).to_csv("mc_test_y.csv")
        pd.DataFrame(prediction).to_csv("mc_prediction.csv")
        print(len(prediction))
        print(len(test_y))
        print(len(train_y))
    # print(len(prediction))
pred_test()











