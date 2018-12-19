# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Using_XGB_CV.py
@time: 2018/2/16 12:34
@Software: PyCharm Community Edition
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
from sklearn import metrics

df = pd.read_csv('Final_version_train.csv',index_col=0)
data = df.iloc[:,:-1].values
label = df.iloc[:,-1].values

data_blocks = np.split(data,10)
label_blocks = np.split(label,10)

def create_split(data_blocks, label_blocks, index):
    num_blocks = len(data_blocks)
    X_val = data_blocks[index]
    Y_val = label_blocks[index]
    if index == 0:
        X_train = np.vstack(data_blocks[1:])
        Y_train = np.hstack(label_blocks[1:])
    elif index == num_blocks-1:
        X_train = np.vstack(data_blocks[:num_blocks-1])
        Y_train = np.hstack(label_blocks[:num_blocks-1])
    else:
        X_train = np.vstack(data_blocks[:index] + data_blocks[index + 1:])
        Y_train = np.hstack(label_blocks[:index] + label_blocks[index + 1:])
    # print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    return X_train, Y_train, X_val, Y_val

filename='log_10.txt'
if os.path.isfile(filename):
    print('File {0} already exists. Deleting file...'.format(filename))
    os.remove(filename)
f=open(filename, mode='x')



num_split = 10


avg_accuracy = np.zeros(8)
std_accuracy = np.zeros(8)
num_max_depth = np.zeros(8)

for i in range(8):
    max_depth = (i+1) * 3
    num_max_depth[i] = max_depth
    accuracy = np.zeros(num_split)
    last_time = time.time()
    for j in range(num_split):
        X_train, Y_train, X_val, Y_val = create_split(data_blocks, label_blocks, j)
        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dtest = xgb.DMatrix(X_val)
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'error',
                  'max_depth': max_depth,
                  'lambda': 10,
                  'subsample': 0.75,
                  'colsample_bytree': 0.75,
                  'min_child_weight': 2,
                  'eta': 0.025,
                  'seed': 0,
                  'silent': 1}
        bst = xgb.train(params, dtrain, num_boost_round=100)
        ypred = bst.predict(dtest)
        prediction = (ypred >= 0.5) * 1
        match = sum(prediction == Y_val)
        accuracy[j] = match / Y_val.shape[0]
        string = "Classification rate: {0}/{1} ({2:.1f}%) with max_depth={3} and split_index={4}"\
            .format(match, Y_val.shape[0], accuracy[j] * 100, max_depth,j)
        print(string)
        f.write(string+'\n')
    avg_accuracy[i] = np.mean(accuracy)
    std_accuracy[i] = np.std(accuracy)
    dashes = "---------------------------------------------------------------------------------------------"
    result = "average classification rate: {0:.1f}% (std={1:.4f}) with max_depth={2}, elapsed time {3:.1f} sec."\
          .format(avg_accuracy[i] * 100, std_accuracy[i], max_depth ,time.time() - last_time)
    print(dashes)
    print(result)
    print(dashes)
    f.write(dashes+'\n')
    f.write(result+'\n')
    f.write(dashes+'\n')
f.close()















