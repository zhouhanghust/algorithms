# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:45:27 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1000)
y = 5 - 0.1*t + 0.5*np.power(t,2) + \
        (6000+5*np.random.randn(len(t)))*np.sin((1+0.005*np.random.randn(len(t)))*t+2) + \
         60*np.random.randn(len(t))

plt.figure()
plt.plot(y)
plt.xlim([0,365])
plt.show()

np.save('mc_data',y)
