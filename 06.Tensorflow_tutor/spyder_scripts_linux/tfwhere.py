# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

a = tf.constant(np.arange(-5,5),dtype=tf.int32)

b = tf.where(a>0)



with tf.Session() as sess:
    print(sess.run(a))
    print(np.squeeze(sess.run(b)))
    print(sess.run(a)[np.squeeze(sess.run(b))])
