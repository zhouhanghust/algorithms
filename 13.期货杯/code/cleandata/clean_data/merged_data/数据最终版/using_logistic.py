#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: using_logistic.py 
@time: 2018/02/{DAY} 
"""
# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: using_svm.py
@time: 2018/2/4 19:21
@Software: PyCharm Community Edition
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv('Final_version_train.csv',index_col=0)
df.index = np.arange(len(df))

X = df.iloc[:,:-1].values
Y = df['y'].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size = 0.25, random_state = 0)




# clf = svm.SVC(C=1.0,kernel='poly',degree=1,coef0=0,probability=True)
clf = LogisticRegression()
# scores = cross_val_score(clf,X,Y,cv=10)
# print(scores.mean())

clf.fit(X_train,y_train)
# print(clf.score(X_test,y_test))
ypred = clf.predict(X_test)

probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
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


from sklearn import metrics
print('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
print('ACC: %.4f' % metrics.accuracy_score(y_test,ypred))
print('Recall: %.4f' % metrics.recall_score(y_test,ypred))
print('F1-score: %.4f' %metrics.f1_score(y_test,ypred))
print('Precesion: %.4f' %metrics.precision_score(y_test,ypred))
print(metrics.confusion_matrix(y_test,ypred))

