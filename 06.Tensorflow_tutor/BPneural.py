# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib import slim
import tensorflow as tf
from sklearn.metrics import accuracy_score

def load_data():
    images = np.load('mnist_images.npy')
    labels = np.load('mnist_labels.npy')
    sample_size = labels.shape[0]
    # data = np.hstack((images.reshape((sample_size, 784)), labels.reshape((sample_size, 1))))
    # np.random.shuffle(data)
    X,Y = shuffle(images.reshape((sample_size,784)),labels.reshape((sample_size,1)))

    X = X.astype(float)
    X_train = X[:50000, :]
    X_test = X[50000:, :]

    # labels=data[:, -1]
    # Y = np.zeros(shape=(sample_size, 10), dtype=float)
    # I = np.eye(10, 10, dtype=float)
    # for i in range(sample_size):
    #     Y[i, :] = I[labels[i], :]
    ohe = OneHotEncoder(categorical_features=[0])
    Y = ohe.fit_transform(Y).toarray()
    Y_train = Y[:50000, :]
    Y_test = Y[50000:, :]

    # Y_train = Y[:50000]
    # Y_test = Y[50000:]

    return X_train, X_test, Y_train, Y_test

X_train,X_test, Y_train, Y_test = load_data()

sc = preprocessing.StandardScaler()
sc.fit(X_train)
batch_size = 10
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# print(X_test[19:20].shape)


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else :
        outputs = activation_function(Wx_plus_b)
    return outputs


X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

lr = tf.Variable(0.001,dtype=tf.float32)

l1 = add_layer(X,784,392,tf.nn.sigmoid)
l2 = add_layer(l1,392,196,tf.nn.sigmoid)
l3 = add_layer(l2,196,49,tf.nn.sigmoid)
l4 = add_layer(l3,49,10)

# preds = tf.nn.softmax(l4)

def train():

    n = X_train.shape[0] // batch_size
    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=l4)
    # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=l4))
    # loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(preds),axis=1))
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)
    init_epoch = tf.Variable(0,dtype=tf.int32,    #trainable=False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(50):
            sess.run(lr.assign(0.001*(0.95**(epoch))))
            for j in range(n):
                input_x = X_train[j*batch_size:(j+1)*batch_size,:]
                input_y = Y_train[j*batch_size:(j+1)*batch_size,:]
                # input_y = Y_train[j * batch_size:(j + 1) * batch_size]
                _,_loss = sess.run([train_op,loss],feed_dict={X:input_x,Y:input_y})

                if (j+1)%100 == 0 :
                    print('Epoch {0} step {1} loss {2:.3f}'.format(epoch+1,j+1,_loss))
            sess.run(init_epoch.assign(epoch+1))
            print("model_save",saver.save(sess,'model_save\\model.ckpt',global_step=epoch+1))

        print("The train has finished!")

# train()


def re_train():
    n = X_train.shape[0] // batch_size
    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=l4)
    # loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(preds),axis=1))
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.minimize(loss)
    init_epoch = tf.Variable(0, dtype=tf.int32)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model_save\\model.ckpt-5')
        initial_num = sess.run(init_epoch)
        for epoch in range(initial_num,50):
            sess.run(lr.assign(0.001*(0.95**(epoch))))
            for j in range(n):
                input_x = X_train[j*batch_size:(j+1)*batch_size,:]
                input_y = Y_train[j*batch_size:(j+1)*batch_size,:]
                # input_y = Y_train[j * batch_size:(j + 1) * batch_size]
                _,_loss = sess.run([train_op,loss],feed_dict={X:input_x,Y:input_y})

                if (j+1)%100 == 0 :
                    print('Epoch {0} step {1} loss {2:.3f}'.format(epoch+1,j+1,_loss))
            sess.run(init_epoch.assign(epoch + 1))
            print("model_save",saver.save(sess,'model_save\\model.ckpt',global_step=epoch+1))

        print("The train has finished!")

# re_train()




def pred():
    saver = tf.train.Saver()
    preds = tf.nn.softmax(l4)
    with tf.Session() as sess:
        saver.restore(sess,'model_save\\model.ckpt-19')
        prediction = sess.run(preds,feed_dict={X:X_test})
        prediction = sess.run(tf.argmax(prediction,axis=1))
        Y_t = np.argmax(Y_test,axis=1)
    print("Accuracy Score:{0:.3f}".format(accuracy_score(prediction,Y_t)))

pred()






















