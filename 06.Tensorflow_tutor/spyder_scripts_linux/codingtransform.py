# -*- coding:utf-8 -*-

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

y = le.fit_transform([3,4,3,4,3,3,4])
print(y)