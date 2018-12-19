# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

df = pd.read_csv('wine.csv',header=0)
X = df[df.columns[1:13]].values
Y = df['Wine'].values
le = LabelEncoder()
Y = le.fit_transform(Y)

# Y = tf.one_hot(indices=y,depth=3,on_value=1.
#                ,off_value=0.,axis=1,name='a')
ohe = OneHotEncoder(categorical_features=[0])
Y = ohe.fit_transform(np.expand_dims(Y,1)).toarray()

X,Y = shuffle(X,Y)

sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)

x = tf.placeholder(tf.float32,[None,12])
W = tf.Variable(tf.truncated_normal([12,3],stddev=0.1))
b = tf.Variable(tf.zeros([3])+0.1)
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32,[None,3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),axis=1))
lr = tf.Variable(0.01,dtype=tf.float32)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

y_pred = tf.arg_max(y,1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    costs = []
    for epoch in range(300):
        sess.run(lr.assign(0.01*(0.95)**epoch))
        sess.run(train_op,feed_dict={x:X,y_:Y})
        costs.append(sess.run(cross_entropy,feed_dict={x:X,y_:Y}))
    ypred = sess.run(y_pred,feed_dict={x:X})

plt.figure()
plt.title("trainingloss lower and lower!")
plt.plot(costs)
plt.show()

y = np.argmax(Y,axis=1)
print('Accuracy: %.2f' % accuracy_score(y,ypred))


