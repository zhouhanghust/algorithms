# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: USE_SVR.py
@time: 2018/2/14 12:29
@Software: PyCharm Community Edition
"""

import scipy.io as sio
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from sklearn.metrics import mean_squared_error

maxmin = pd.read_csv('maxmin.csv',index_col=0)
min_ZDJ = maxmin.iloc[1,0]
max_ZDJ = maxmin.iloc[0,0]

data = sio.loadmat('ShangZZS.mat')
X_train = data['X_train']
X_train = X_train[::-1,:]
X_test = data['X_test']
X_test = X_test[::-1,:]
y_train = data['y_train'][:,0]
y_train = y_train[::-1]
y_test = data['y_test'][:,0]
y_test = y_test[::-1]

#############################################################################
# 训练SVR模型
# 初始化SVR
svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-9, 9, 20)},
                   scoring='neg_mean_squared_error')

# 训练
svr.fit(X_train, y_train)

print("svr_best_params:",svr.best_params_)
train_result = svr.predict(X_train)
train_result = train_result*(max_ZDJ-min_ZDJ) + min_ZDJ
# 测试
test_result = svr.predict(X_test)
test_result = test_result*(max_ZDJ-min_ZDJ) + min_ZDJ

y_train = y_train*(max_ZDJ-min_ZDJ) + min_ZDJ
y_test = y_test*(max_ZDJ-min_ZDJ) + min_ZDJ


# mse = np.mean((test_result - y_test)**2)
mse = mean_squared_error(y_test,test_result)
print("MSE:",mse)

rmse = np.sqrt(mse)
print('RMSE:',rmse)
mape = np.abs((y_test-test_result)/y_test)
mape = np.mean(mape)
print('MAPE:',mape)




#############################################################################
# 对结果进行显示
y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

plt.figure()

font = FontProperties(fname="simhei.ttf", size=16)

plt.plot(np.arange(len(y_train)+len(y_test)),np.vstack([y_train,y_test]), c='k',
         label='真实值',linewidth=2.0)
# plt.plot(np.arange(len(y_train)), train_result, c='r',
#          label='data_SVR ')
# plt.plot(np.arange(len(y_train),len(y_train)+len(y_test)), test_result, c='r',
#          label='data_SVR ')
plt.plot(np.arange(len(y_train)+len(y_test)), np.hstack([train_result,test_result]), c='r',
         label='SVR预测值',linestyle='--',linewidth=2.0,marker='+',markersize=13)
plt.xlabel('日期（2017/12/12-2018/02/09）',fontproperties=font)
plt.ylabel('上证指数最低价',fontproperties=font)
# plt.title('SVR')
plt.legend(prop=font)
plt.show()

# sio.savemat('SVR.mat',{'pred':np.hstack([train_result,test_result])})

