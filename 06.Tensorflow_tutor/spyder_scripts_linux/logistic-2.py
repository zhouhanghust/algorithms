# -*- coding:utf-8 -*-

import tensorflow.contrib.learn as skflow
from sklearn import datasets,metrics,preprocessing
import numpy as np
import pandas as pd

df = pd.read_csv("CHD.csv",header=0)

def my_model(X,y):
    return skflow.models.logistic_regression(X,y)
a = preprocessing.StandardScaler()
X1 = a.fit_transform(df['age'].values.astype(np.float32))
y1 = df['chd'].values

print(X1)















