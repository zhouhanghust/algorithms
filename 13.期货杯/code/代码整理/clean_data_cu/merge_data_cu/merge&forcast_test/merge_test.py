# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: merge_test.py
@time: 2018/2/5 14:57
@Software: PyCharm Community Edition
"""

import numpy as np
import pandas as pd

df1 = pd.read_csv("test.csv",index_col=0,header=0)
df2 = pd.read_csv("merged_DATA_V2.csv",index_col=0,header=0)

# print(df1.columns)
# print(df2.columns)

dfout = pd.merge(df1,df2,on='report_id',how='left')
# print(dfout.columns)

dfout.to_csv("merged_testV1.csv")