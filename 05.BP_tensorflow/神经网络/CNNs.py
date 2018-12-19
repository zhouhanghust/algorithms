"""
Convolutional neural networks
"""

import tensorflow as tf
from tensorflow.contrib import slim
from utils import *
import time

batch_size = 10

x = tf.placeholder(shape=(batch_size, 28, 28, 1), dtype=tf.float32)
y = tf.placeholder(shape=(batch_size, 10), dtype=tf.int64)

m, v = tf.nn.moments(x, axes=[0, 1, 2, 3], keep_dims=True)
normalized_x = tf.div(x-m, tf.sqrt(v))

conv1 = slim.conv2d(normalized_x, num_outputs=16, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
print('conv1 shape:', conv1.get_shape())

pool1 = slim.max_pool2d(conv1, kernel_size=[2, 2])
print('pool1 shape:', pool1.get_shape())

conv2 = slim.conv2d(pool1, num_outputs=32, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
print('conv2 shape:', conv2.get_shape())

pool2 = slim.max_pool2d(conv2, kernel_size=[2, 2])
print('pool2 shape:', pool2.get_shape())

flatten = slim.flatten(pool2)
print('flatten shape:', flatten.get_shape())
fc = slim.fully_connected(flatten, 32)
logits = slim.fully_connected(fc, 10, activation_fn=None)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

match = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_sum(tf.cast(match, tf.float32))/batch_size

opt = tf.train.GradientDescentOptimizer(0.01)
train_op = opt.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    X_train, X_test, Y_train, Y_test = load_data4D()
    n1 = X_train.shape[0] // batch_size
    n2 = X_test.shape[0] // batch_size

    sess.run(init)
    for i in range(10):
        lt=time.time()
        for j in range(n1):
            input_x = X_train[j * batch_size: (j + 1) * batch_size, :]
            input_y = Y_train[j * batch_size: (j + 1) * batch_size, :]
            _, _loss = sess.run([train_op, loss], feed_dict={x:input_x, y:input_y})

            if (j+1)%100==0:
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
