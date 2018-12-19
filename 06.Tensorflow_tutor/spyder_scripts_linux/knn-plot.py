#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 11:41:10 2017

@author: zhouhang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.Session() as sess:
    fig,ax = plt.subplots()
    ax.plot(tf.random_normal([100]).eval(),tf.random_normal([100]).eval(),'o')
    ax.set_title('Sample random plot for Tensorflow')
    plt.savefig("result.png")