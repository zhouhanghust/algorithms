# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets,cross_validation,metrics
from sklearn import preprocessing
import tensorflow as tf
import numpy as np

df = pd.read_csv("mpg.csv",header=0)
df['displacement'] = df['displacement'].astype(np.float32)

X = df[df.columns[1:8]].values
y = df['mpg'].values

# plt.figure()
# for i in range(1,8):
#     number = 420 + i
#     ax1 = plt.subplot(number)
#     ax1.locator_params(nbins=3)
#     plt.title(list(df)[i])
#     ax1.scatter(df[df.columns[i]],y)
# plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
# plt.show()

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25,random_state=12)

scaler = preprocessing.StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def addLayer(inputData,inSize,outSize,activity_function = None):
    Weights = tf.Variable(tf.truncated_normal([inSize,outSize],stddev=0.1))
    bias = tf.Variable(tf.zeros([outSize])+0.1)
    weights_plus_b = tf.matmul(inputData,Weights)+bias
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


XS = tf.placeholder(tf.float32, [None, 7])
YS = tf.placeholder(tf.float32, [None])

l1 = addLayer(XS, 7, 12, activity_function=tf.nn.sigmoid)
l2 = addLayer(l1, 12, 6, activity_function=tf.nn.sigmoid)
l3 = addLayer(l2,6,1,activity_function=None)

loss = tf.reduce_mean(tf.pow(YS - tf.squeeze(l3),2))
lr = tf.Variable(0.01,dtype=tf.float32)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    costs = []
    sess.run(init)
    for epoch in range(500):
        sess.run(lr.assign(0.01*(0.95)**epoch))
        sess.run(train_op,feed_dict={XS:X_train,YS:y_train})
        costs.append(sess.run(loss,feed_dict={XS:X_test,YS:y_test}))
    print(sess.run(l3,feed_dict={XS:X_test,YS:y_test}).shape)
plt.figure()
plt.plot(costs)
plt.show()

















