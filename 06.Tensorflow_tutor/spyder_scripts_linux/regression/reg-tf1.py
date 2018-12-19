# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

trX = np.linspace(-1,1,101)
trY = 2*trX + np.random.randn(trX.shape[0])*0.4 + 0.2

# fig,ax = plt.subplots()
# ax.scatter(trX,trY)
# ax.plot(trX,.2+2*trX)
# plt.show()

X = tf.placeholder(dtype=tf.float32,name="X")
Y = tf.placeholder(dtype=tf.float32,name="Y")

with tf.name_scope("Model"):
    def model(X,w,b):
        return tf.multiply(X,w)+b
    w = tf.Variable(-1.0,name='b0')
    b = tf.Variable(-2.0,name='b1')
    y_model = model(X,w,b)

with tf.name_scope('CostFunction'):
    cost = (tf.pow(Y-y_model,2))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_1 = []
    for i in range(100):
        for (x,y) in zip(trX,trY):
            sess.run(train_op,feed_dict={X:x,Y:y})
    for x in trX:
        y_1.append(sess.run(y_model,feed_dict={X:x}))
    [w_1,b_1] = sess.run([w,b])

    plt.plot(trX,y_1)
    plt.scatter(trX,trY)
    plt.show()
