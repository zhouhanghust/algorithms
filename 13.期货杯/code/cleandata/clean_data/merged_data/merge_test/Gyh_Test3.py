# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Gyh_Test3.py
@time: 2018/2/9 20:37
@Software: PyCharm Community Edition
"""

import pandas as pd
import numpy as np

dftest = pd.read_csv("after_GYH_test.csv",index_col=0)

ind1 = dftest['djk_latest6_month_used_avg_sumamount/djk_used_credit_limit_sumamount'].notnull()
dftest['djk_latest6_month_used_avg_sumamount/djk_used_credit_limit_sumamount'][-ind1] = 0

ind2 = dftest['djkyq_highest_oa_per_mon/djk_credit_limit_sumamount'] < 0
dftest['djkyq_highest_oa_per_mon/djk_credit_limit_sumamount'][ind2] = 0

dftest.to_csv("after_GYH_testV2.csv")


