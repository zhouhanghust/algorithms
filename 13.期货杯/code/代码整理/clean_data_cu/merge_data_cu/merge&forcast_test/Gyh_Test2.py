# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Gyh_Test2.py
@time: 2018/2/9 20:16
@Software: PyCharm Community Edition
"""

import numpy as np
import pandas as pd

df = pd.read_csv("before_GYH_test.csv")

ind1 = df['dk_state'] < 0
df['dk_state'][ind1] = 0

ind3 = df['dk_ratio24'] < 0
df['dk_ratio24'][ind3] = 0

ind4 = df['djk_used_credit_limit_sumamount/djk_credit_limit_sumamount'] > 1
df['djk_used_credit_limit_sumamount/djk_credit_limit_sumamount'][ind4]=1

list1 = ['djk_latest6_month_used_avg_sumamount/djk_used_credit_limit_sumamount',
        'djk_used_highest_maxamount/djk_credit_limit_sumamount',
        'djkyq_highest_oa_per_mon/djk_credit_limit_sumamount',
        'dk_latest_6m_debt_ratio']
len(list1)
for each in list1:
    ind = df[each] > 1
    df[each][ind] = 1

del list1

ind = df['dk_latest_6m_debt_ratio'] < 0
df['dk_latest_6m_debt_ratio'][ind] = 0


guiyi_list = ['edu_level', 'salary',
       'djkyq_count_dw', 'djkyq_months',
       'djkyq_max_duration', 'wjqdjk_finance_corp_count',
       'wjqdjk_finance_org_count', 'wjqdjk_account_count',
       'wjqdjk_credit_limit', 'wjqdjk_max_credit_limit_per_org',
       'wjqdjk_min_credit_limit_per_org',
       'dkyq_max_duration',
       'wjqdk_finance_corp_count', 'wjqdk_finance_org_count',
       'wjqdk_account_count', 'wjqdk_credit_limit',
       'wjqdk_latest_6m_used_avg_amount', 'cre_house_loan_count',
       'cre_commercial_loan_count', 'cre_other_loan_count',
       'query_times',
       'cre_card_count']

for each in guiyi_list:
    df[each] = (df[each]-df[each].min())/(df[each].max()-df[each].min())


ind = df['djk_last_overdue']<0
df['djk_last_overdue'][ind] = 0

ind = df['has_fund']>0.5
df['has_fund'][ind]=1
df['has_fund'][-ind]=0

ind = df['djk_last_overdue'] > 0.5
df['djk_last_overdue'][ind] = 1
df['djk_last_overdue'][-ind] = 0

# ind = df['dk_state']>0.5
# df['dk_state'][ind] = 1
# df['dk_state'][-ind] = 0

df['dk_state'] = df['dk_state'].map(np.round)

# print(df.head(2))

df.to_csv("after_GYH_test.csv")



