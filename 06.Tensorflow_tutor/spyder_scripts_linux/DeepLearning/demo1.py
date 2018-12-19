# -*- coding:utf-8 -*-
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

img = misc.imread("data1/test.gif")
plt.imshow(img[:,:,0])
plt.show()