#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: Plot_maxdepth.py 
@time: 2018/02/{DAY} 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data = np.array([0.8,0.858,0.898,0.934,0.95,0.954,0.955,0.956])
labels = [3,6,9,12,15,18,21,24,27]

plt.figure()
font = FontProperties(fname="simhei.ttf", size=16)

plt.plot(data,linewidth=2.0)
plt.xticks(np.arange(8),labels,fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('max_depth',fontproperties=font,fontsize=40)
plt.ylabel('分类准确率',fontproperties=font,fontsize=30)
plt.show()





