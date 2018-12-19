# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Using_RF.py
@time: 2018/2/16 11:54
@Software: PyCharm Community Edition
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import time

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


num_split = 10
max_depth = 12

filename='log_10.txt'
if os.path.isfile(filename):
    print('File {0} already exists. Deleting file...'.format(filename))
    os.remove(filename)
f=open(filename, mode='x')

avg_accuracy = np.zeros(10)
std_accuracy = np.zeros(10)
num_estimators = np.zeros(10, dtype=int)
for i in range(10):
    n = (i + 1) * 9
    num_estimators[i]=n
    accuracy = np.zeros(num_split)
    last_time = time.time()
    for j in range(num_split):
        X_train, Y_train, X_val, Y_val = create_split(data_blocks, label_blocks, j)
        classifier = RandomForestClassifier(n_estimators=n, max_depth=max_depth, n_jobs=-1)
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_val)
        match = sum(prediction == Y_val)
        accuracy[j] = match / Y_val.shape[0]
        string = "Classification rate: {0}/{1} ({2:.1f}%) with n_estimators={3} and split_index={4}"\
            .format(match, Y_val.shape[0], accuracy[j] * 100, n, j)
        print(string)
        f.write(string+'\n')
    avg_accuracy[i] = np.mean(accuracy)
    std_accuracy[i] = np.std(accuracy)
    dashes = "---------------------------------------------------------------------------------------------"
    result = "n_estimators={2}: average classification rate: {0:.1f}% (std={1:.4f}), elapsed time {3:.1f} sec."\
          .format(avg_accuracy[i] * 100, std_accuracy[i], n, time.time() - last_time)
    print(dashes)
    print(result)
    print(dashes)
    f.write(dashes+'\n')
    f.write(result+'\n')
    f.write(dashes+'\n')
f.close()
# np.save('log_avg_{0}.npy'.format(max_depth), avg_accuracy)
# np.save('log_std_{0}.npy'.format(max_depth), std_accuracy)
# np.save('log_num_{0}.npy'.format(max_depth), num_estimators)