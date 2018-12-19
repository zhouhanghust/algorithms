#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: run_xgb.py 
@time: 2018/02/{DAY} 
"""

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


df = pd.read_csv("Final_version_train.csv",index_col=0)
df_test = pd.read_csv("after_GYH_testV2.csv",index_col=0)
df_test = df_test[['report_id']+df.columns[:-1].tolist()]
data_test = df_test.iloc[:,1:].values

data = df.iloc[:,:-1].values
label = df.iloc[:,-1].values

# train_x, test_x, train_y, test_y = train_test_split(data, label,test_size=0.25,random_state=0)
# print(train_x.shape,test_y.shape)

dtrain=xgb.DMatrix(data,label=label)
# dtest=xgb.DMatrix(test_x,label=test_y)
dtest=xgb.DMatrix(data_test)

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

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1


result = pd.DataFrame({'report_id':df_test.iloc[:,0].values,'y_pred':y_pred})

# print(result)

result.to_csv('prediction.csv')

# from sklearn import metrics
# print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
# print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
# print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
# print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
# print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
# print(metrics.confusion_matrix(test_y,y_pred))


# xgb.plot_importance(bst)
# plt.show()