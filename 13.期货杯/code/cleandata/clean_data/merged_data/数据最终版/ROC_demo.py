#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: ROC_demo.py 
@time: 2018/02/{DAY} 
"""

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
###############################################################################
# Data IO and generation,导入iris数据，做数据准备

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]  # 去掉了label为2，label只能二分，才可以。
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
###############################################################################
# Classification and ROC analysis
# 分类，做ROC分析

# Run classifier with cross-validation and plot ROC curves
# 使用6折交叉验证，并且画ROC曲线

classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)  # 注意这里，probability=True,需要，不然预测的时候会出现异常。另外rbf核效果更好些。

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []


# 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
#    print set(y[train])                     #set([0,1]) 即label有两个类别
#    print len(X[train]),len(X[test])        #训练集有84个，测试集有16个
#    print "++",probas_                      #predict_proba()函数输出的是测试集在lael各类别上的置信度，
#    #在哪个类别上的置信度高，则分为哪类
# Compute ROC curve and area the curve
# 通过roc_curve()函数，求出fpr和tpr，以及阈值
# print(probas_)
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