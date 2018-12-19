#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: Clean_data_code2.py 
@time: 2018/02/{DAY} 
"""
import pandas as pd
import numpy as np

djk1 = pd.read_csv('djk1.csv',index_col=0)
djk2 = pd.read_csv('djk2.csv',index_col=0)

djk3 = pd.read_csv('djk3.csv',index_col=0)
djk4 = pd.read_csv('djk4.csv',index_col=0)

dk1 = pd.read_csv('dk1.csv',index_col=0)
dk2 = pd.read_csv('dk2.csv',index_col=0)
dk3 = pd.read_csv('dk3.csv',index_col=0)

credicue = pd.read_csv('credicue.csv',index_col=0)
recordsmr = pd.read_csv('recordsmr.csv',index_col=0)

df = pd.merge(djk1,djk2,on='report_id',how='outer')
df = pd.merge(df,djk3,on='report_id',how='outer')
df = pd.merge(df,djk4,on='report_id',how='outer')
df = pd.merge(df,dk1,on='report_id',how='outer')
df = pd.merge(df,dk2,on='report_id',how='outer')
df = pd.merge(df,dk3,on='report_id',how='outer')
df = pd.merge(df,credicue,on='report_id',how='outer')
df = pd.merge(df,recordsmr,on='report_id',how='outer')




dellist = ['cre_first_sl_open_month','djk_currency','djk_state','dkyq_count_dw','dkyq_months','wjqdjk_balance','wjqdk_used_credit_limit']
for each in dellist:
    del df[each]

df.rename(columns={'djk_credit_limit_amount':'djk_credit_limit_sumamount',
'djk_latest6_month_used_avg_amount':'djk_latest6_month_used_avg_sumamount',
'djk_payment_state':'djk_ratio24',
'djk_used_credit_limit_amount':'djk_used_credit_limit_sumamount',
'djk_used_highest_amount':'djk_used_highest_maxamount',
'djkyq_amount':'djk_yqamount','querier':'query_times',
'djkyq_last_months':'djk_last_months'
},inplace=True)

# print(df.columns)

df.to_csv('merge_data_cu/merged_data.csv')

