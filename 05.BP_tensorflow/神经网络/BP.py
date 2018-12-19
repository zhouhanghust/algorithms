"""
vanilla neural networks
"""

import numpy as np
import tensorflow as tf
from utils import load_data
import time

batch_size = 10

x = tf.placeholder(shape=(batch_size, 784), dtype=tf.float32)
y = tf.placeholder(shape=(batch_size, 10), dtype = tf.int64)

m, v = tf.nn.moments(x, axes=1, keep_dims=True)
normalized_x = tf.div(x-m, tf.sqrt(v))

W1 = tf.Variable(initial_value=np.random.randn(784, 30)/np.sqrt(784), dtype=tf.float32)
b1 = tf.Variable(initial_value=np.zeros((30,), dtype=np.float32), dtype=tf.float32)

W2 = tf.Variable(initial_value=np.random.randn(30, 10)/np.sqrt(30), dtype=tf.float32)
b2 = tf.Variable(initial_value=np.zeros((10,), dtype=np.float32), dtype=tf.float32)

activation1 = tf.nn.sigmoid(tf.matmul(normalized_x, W1) + b1)
activation2 = tf.matmul(activation1, W2) + b2

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=activation2)

match = tf.equal(tf.argmax(activation2, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_sum(tf.cast(match, tf.float32))/batch_size

opt = tf.train.GradientDescentOptimizer(0.01)
train_op = opt.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    X_train, X_test, Y_train, Y_test = load_data()
    n1 = X_train.shape[0] // batch_size
    n2 = X_test.shape[0] // batch_size

    sess.run(init)
    for i in range(20):
        lt=time.time()
        for j in range(n1):
            input_x = X_train[j * batch_size: (j + 1) * batch_size, :]
            input_y = Y_train[j * batch_size: (j + 1) * batch_size, :]
            _, _loss = sess.run([train_op, loss], feed_dict={x:input_x, y:input_y})
            if (j+1)%500==0:
                print('Epoch {0} step {1} loss {2:.3f}'.format(i+1, j+1, _loss))

        test_accuracy = np.zeros(n2)
        test_loss = np.zeros(n2)
        for j in range(n2):
            input_x = X_test[j * batch_size: (j + 1) * batch_size, :]
            input_y = Y_test[j * batch_size: (j + 1) * batch_size, :]
            test_accuracy[j], test_loss[j] = sess.run([accuracy, loss], feed_dict={x: input_x, y: input_y})
        print("=======================================================")
        print('Epoch {0}: ({3:.1f} sec) test loss {1:.3f} test accuracy {2:.3f}%'\
              .format(i+1, np.mean(test_loss), 100*np.mean(test_accuracy), time.time()-lt))
        print("=======================================================")
