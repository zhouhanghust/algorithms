# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Gyh_Test.py
@time: 2018/2/9 12:21
@Software: PyCharm Community Edition
"""

import pandas as pd
import numpy as np

dftest = pd.read_csv("testbu.csv",index_col=0)

dftrain = pd.read_csv("Final_version_train.csv",index_col=0)

# print(dftrain.columns.values)
# print(set(dftest.columns))

dftest['djk_latest6_month_used_avg_sumamount/djk_used_credit_limit_sumamount'] = \
    dftest['djk_latest6_month_used_avg_sumamount']/dftest['djk_used_credit_limit_sumamount']

dftest['djk_used_highest_maxamount/djk_credit_limit_sumamount'] = \
    dftest['djk_used_highest_maxamount']/dftest['djk_credit_limit_sumamount']

dftest['cre_card_count'] = \
    dftest['cre_standard_loancard_count']+dftest['cre_loancard_count']

dftest['djk_used_credit_limit_sumamount/djk_credit_limit_sumamount'] = \
    dftest['djk_used_credit_limit_sumamount']/dftest['djk_credit_limit_sumamount']

dftest['djkyq_highest_oa_per_mon/djk_credit_limit_sumamount'] = \
    dftest['djkyq_highest_oa_per_mon']/dftest['djk_credit_limit_sumamount']
dftest['dk_latest_6m_debt_ratio'] = \
    dftest['wjqdk_latest_6m_used_avg_amount']/dftest['wjqdk_credit_limit']

set1 = set(dftrain.columns)
set2 = set(dftest.columns)

set_jiaoji = set1&set2
set_train_unique = set1 - set_jiaoji
set_test_unique = set2 - set_jiaoji

# print(set_train_unique)
# print(set_test_unique)

for each in set_test_unique:
    del dftest[each]

set1 = set(dftrain.columns)
set2 = set(dftest.columns)
set_jiaoji1 = set1&set2
set_train_unique1 = set1 - set_jiaoji
set_test_unique1 = set2 - set_jiaoji

print("================after modifing=========================")
print(set_train_unique1)
print(set_test_unique1)

dftest.to_csv("before_GYH_test.csv")