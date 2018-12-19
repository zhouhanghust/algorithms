# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Mer_Plot.py
@time: 2018/2/15 17:13
@Software: PyCharm Community Edition
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data_GNN = sio.loadmat('GA_GNN.mat')
y_true = data_GNN['y_true']
y_GNN = data_GNN['pred']

data_BP = sio.loadmat('BP.mat')
y_BP = data_BP['pred']

data_SVR = sio.loadmat('SVR.mat')
y_SVR = data_SVR['pred']

plt.figure()
font = FontProperties(fname="simhei.ttf", size=16)
font2 = FontProperties(fname="simhei.ttf", size=21)
plt.plot(np.arange(40),np.squeeze(y_true), c='k',
         label='真实值',linewidth=2.0)

plt.plot(np.arange(40), np.squeeze(y_BP), c='yellowgreen',
         label='BP预测值',linestyle='--',linewidth=2.0,marker='+',markersize=13)

plt.plot(np.arange(40), np.squeeze(y_SVR), c='blue',
         label='SVR预测值',linestyle='-.',linewidth=2.0,marker='v',markersize=11)

plt.plot(np.arange(40), np.squeeze(y_GNN), c='r',
         label='GA_GNN预测值',linestyle='--',linewidth=2.0,marker='*',markersize=13)

plt.xlabel('日期（2017/12/12-2018/02/09）',fontproperties=font,fontsize=25)
plt.ylabel('上证指数最低价',fontproperties=font,fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop=font2)
plt.show()






















