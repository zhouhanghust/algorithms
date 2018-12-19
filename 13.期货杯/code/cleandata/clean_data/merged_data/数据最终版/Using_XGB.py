# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Using_XGB.py
@time: 2018/2/13 20:45
@Software: PyCharm Community Edition
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("Final_version_train.csv",index_col=0)
data = df.iloc[:,:-1].values
label = df.iloc[:,-1].values

train_x, test_x, train_y, test_y = train_test_split(data, label,test_size=0.25,random_state=0)
# print(train_x.shape,test_y.shape)

dtrain=xgb.DMatrix(train_x,label=train_y)
# dtest=xgb.DMatrix(test_x,label=test_y)
dtest=xgb.DMatrix(test_x)

params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':15,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

# watchlist = [(dtest,'eval'),(dtrain,'train')]
# watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100)
# bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
ypred=bst.predict(dtest)

fpr, tpr, thresholds = roc_curve(test_y,ypred)
roc_auc = auc(fpr, tpr)
# 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

# 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()










# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

from sklearn import metrics
print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print(metrics.confusion_matrix(test_y,y_pred))
#
#
# xgb.plot_importance(bst)
# plt.show()
