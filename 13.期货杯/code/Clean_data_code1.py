# -*- coding: UTF-8 -*-
"""
@author:
contact:
@file: Clean_data_code1.py
@time: 2018/2/17 20:36
@Software: PyCharm Community Edition
"""
import pandas as pd
import numpy as np
import os
import re
import time
from sklearn.preprocessing import LabelEncoder


contest_ext_crd_cd_ln = pd.read_table('contest_ext_crd_cd_ln.tsv')
# 357196,22
df1 = contest_ext_crd_cd_ln
index = df1['class5_state'].notnull()
df1 = df1[index]

mapdict = {
    "次级": 1,
    "可疑": 1,
    "未知": 0,
    "关注": 0,
    "正常": 0}
temp = df1['class5_state'].map(mapdict)
del df1['class5_state']
pp = ['loan_id', 'curr_overdue_cyc', 'curr_overdue_amount', 'recent_pay_date', 'scheduled_payment_date', 'open_date',
      'end_date', 'remain_payment_cyc', 'payment_cyc', 'finance_org', 'type_dw', 'currency', 'guarantee_type',
      'payment_rating']
for each in pp:
    del df1[each]

df1['class5_state'] = temp
mapdict2 = {'正常': 0, '逾期': 1}
df1['state'] = df1['state'].map(mapdict2)


def payment_state_ratio(lines):
    numsum = 0
    total = 0
    for line in lines:
        line = line.lstrip('/')
        numstr = re.sub("\D", "", line)
        total += len(line) - len(numstr)
        for each in numstr:
            numsum += int(each)
            total += int(each)
    return numsum / total
#
#

a1 = df1['payment_state'].groupby(df1['report_id'])
ratio24 = a1.apply(payment_state_ratio)
ratio24 = ratio24.reset_index()

a2 = df1['scheduled_payment_amount'] > df1['actual_payment_amount']
last_overdue = a2.map(int).groupby(df1['report_id']).max()
last_overdue = last_overdue.reset_index()

a3 = df1[['credit_limit_amount', 'balance']].groupby(df1['report_id']).sum()
debt_ratio = a3['balance'] / a3['credit_limit_amount']
debt_ratio = debt_ratio.reset_index()

pp2 = ['payment_state','scheduled_payment_amount','actual_payment_amount','credit_limit_amount','balance']
for each in pp2:
    del df1[each]

df1 = df1.groupby('report_id').apply(max).reset_index(drop=True)

temp_merge1 = pd.merge(df1, debt_ratio, on='report_id')
temp_merge1 = pd.merge(temp_merge1, last_overdue, on='report_id')
temp_merge1 = pd.merge(temp_merge1, ratio24, on='report_id')

temp_merge1.columns = ['report_id', 'state', 'class5_state', 'debt_ratio', 'last_overdue', 'payment_state']
temp_merge1.rename(columns={'payment_state': 'ratio24'}, inplace=True)

temp_merge1.rename(columns=lambda x:'dk_'+x,inplace=True)
temp_merge1.columns = ['report_id']+temp_merge1.columns.tolist()[1:]
temp_merge1.to_csv('clean_data_cu/dk1.csv')

#=============================================================

contest_ext_crd_cd_lnd = pd.read_table('contest_ext_crd_cd_lnd.tsv')
#324229,20
df2 = contest_ext_crd_cd_lnd
df2['report_id']=df2['report_id'].astype(int)

index = df2['payment_state'].notnull()
df2 = df2[index]

mapdict = {'正常':0,'冻结':1,'止付':1}
df2['state'] = df2['state'].map(mapdict)

delname = ['loancard_id','finance_org','open_date','guarantee_type','scheduled_payment_date','recent_pay_date','curr_overdue_cyc','curr_overdue_amount','cardtype','share_credit_limit_amount']
for each in delname:
    del df2[each]

mapdict = {'人民币':1.00, '加拿大元':5.142, '日元':0.058, '欧元':7.867, '港元':0.810, '澳大利亚元':5.120, '澳门元':0.787, '瑞士法郎':6.712, '美元':6.33, '英镑':9.046}
df2['currency'] = df2['currency'].map(mapdict)

multiplyname = ['credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount','used_highest_amount','scheduled_payment_amount','actual_payment_amount']
for each in multiplyname:
    df2[each] = df2[each] * df2['currency']

ind = df2['currency'] != 1

x = df2[ind]
ind2 = x['used_credit_limit_amount'] != 0.0
len(x[['used_credit_limit_amount']][ind2]) #212

ind = df2['currency'] != 1.0
df2 = df2[-ind]

aa = df2['payment_state'].groupby(df2['report_id'])
df2_ratio24 = aa.apply(payment_state_ratio)
df2_ratio24 = df2_ratio24.reset_index()

temp3 = df2.groupby('report_id')[['credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount']].sum()
temp3 = temp3.reset_index()
highest = df2[['used_highest_amount']].groupby(df2['report_id']).max()
highest = highest.reset_index()

onemonth = df2['scheduled_payment_amount']>df2['actual_payment_amount']
last_overdue = onemonth.map(int).groupby(df2['report_id']).max()
last_overdue = last_overdue.reset_index()
last_overdue.columns = ['report_id','last_overdue']

pp = ['payment_state','scheduled_payment_amount','actual_payment_amount','credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount','used_highest_amount']
for each in pp:
    del df2[each]
df2 = df2.groupby('report_id').apply(max).reset_index(drop=True)
df2['report_id'] = df2['report_id'].astype(int)
temp3 = pd.merge(temp3,highest,on='report_id')
temp3 = pd.merge(temp3,last_overdue,on='report_id')
temp3 = pd.merge(temp3,df2_ratio24,on='report_id')
# print(temp3.head(2))
df2 = pd.merge(df2,temp3,on='report_id')
df2.rename(columns=lambda x:'djk_'+x,inplace=True)
df2.columns = ['report_id']+df2.columns.tolist()[1:]
df2.to_csv('clean_data_cu/djk1.csv')
#==================================================================

contest_ext_crd_cd_lnd_ovd = pd.read_csv('contest_ext_crd_cd_lnd_ovd.csv')
#199644,4
df3 = contest_ext_crd_cd_lnd_ovd
df3 = df3.groupby('REPORT_ID')['LAST_MONTHS','AMOUNT'].sum().reset_index()
df3.rename(columns=lambda x:x.lower(),inplace=True)
df3.rename(columns=lambda x:'djkyq_'+x,inplace=True)
df3.columns = ['report_id']+df3.columns.tolist()[1:]
df3.to_csv('clean_data_cu/djk2.csv')
#=================================================================

contest_ext_crd_is_creditcue = pd.read_csv('contest_ext_crd_is_creditcue.csv')
# 39970,11
dff = contest_ext_crd_is_creditcue

dff.rename(columns=lambda x: x.lower(), inplace=True)

delname = ['first_sl_open_month']

def time2num(timestr):
    temp = time.strptime(timestr, "%Y.%m")
    timeseconds = time.mktime(temp)
    return timeseconds


ind1 = dff['first_loan_open_month'] == '--'
dff['first_loan_open_month'][ind1] = dff[ind1]['first_loancard_open_month'].values

ind2 = dff['first_loancard_open_month'] == '--'
dff['first_loancard_open_month'][ind2] = dff[ind2]['first_loan_open_month']

temp = (dff[['first_loan_open_month']] == '--').join(dff[['first_loancard_open_month']] == '--')
temp.apply(np.all, axis=1)
sum(temp.apply(np.all, axis=1))
ind3 = temp.apply(np.all, axis=1)

dff['first_open_time'] = np.nan
dff['first_open_time'][-ind3] = dff[-ind3][['first_loancard_open_month', 'first_loan_open_month']].applymap(
    time2num).apply(min, axis=1)

def maxmin(x):
    a = max(x)
    b = min(x)
    result = []
    for each in x:
        temp = (each - b) / (a - b)
        result.append(temp)
    return result

dff['first_open_time'][-ind3] = (dff[['first_open_time']][-ind3].apply(maxmin, axis=0)).values
del dff['first_loan_open_month']
del dff['first_loancard_open_month']

dff.rename(columns=lambda x:'cre_'+x,inplace=True)
dff.columns = ['report_id']+dff.columns.tolist()[1:]
dff.to_csv('clean_data_cu/credicue.csv')
#=====================================================

contest_ext_crd_is_ovdsummary = pd.read_csv('contest_ext_crd_is_ovdsummary.csv')
#76212,6
df4 = contest_ext_crd_is_ovdsummary

df4.rename(columns=lambda x:x.lower(),inplace=True)
ind1 = df4['type_dw'] == '贷款逾期'
df41 = df4[ind1]
df42 = df4[-ind1]

df41 = df41.groupby('report_id')['count_dw','months','highest_oa_per_mon','max_duration'].max().reset_index()
df42 = df42.groupby('report_id')['count_dw','months','highest_oa_per_mon','max_duration'].max().reset_index()
df41.rename(columns=lambda x:'dkyq_'+x,inplace=True)
df41.columns = ['report_id']+df41.columns.tolist()[1:]
df42.rename(columns=lambda x:'djkyq_'+x,inplace=True)
df42.columns = ['report_id']+df42.columns.tolist()[1:]
df41.to_csv('clean_data_cu/dk2.csv')
df42.to_csv('clean_data_cu/djk3.csv')

#=================================================================

contest_ext_crd_is_sharedebt = pd.read_csv('contest_ext_crd_is_sharedebt.csv')
#76246,11
df5 = contest_ext_crd_is_sharedebt.copy()
df5.rename(columns=lambda x:x.lower(),inplace=True)
ind = df5['type_dw'] == '未结清贷款信息汇总'
djk = df5[['report_id','max_credit_limit_per_org','min_credit_limit_per_org']][-ind]
djk = djk.groupby('report_id').agg({'max_credit_limit_per_org':max,'min_credit_limit_per_org':min})
djk = djk.reset_index()

df55 = contest_ext_crd_is_sharedebt.copy()
df55 = df55.groupby('REPORT_ID')['CREDIT_LIMIT','BALANCE','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT','FINANCE_CORP_COUNT','FINANCE_ORG_COUNT','ACCOUNT_COUNT'].sum().reset_index()
df55.rename(columns=lambda x:x.lower(),inplace=True)
ind = df5['type_dw'] == '未结清贷款信息汇总'
df51 = df55[ind]
df52 = df55[-ind]
df52 = pd.merge(df52,djk,on='report_id',how='outer')
df51.rename(columns=lambda x:'wjqdk_'+x,inplace=True)
df51.columns = ['report_id']+df51.columns.tolist()[1:]
df52.rename(columns=lambda x:'wjqdjk_'+x,inplace=True)
df52.columns = ['report_id']+df52.columns.tolist()[1:]
df51.to_csv('clean_data_cu/dk3.csv')
df52.to_csv('clean_data_cu/djk4.csv')
#==========================================

contest_ext_crd_qr_recorddtlinfo = pd.read_table('contest_ext_crd_qr_recorddtlinfo.tsv')
df6 = contest_ext_crd_qr_recorddtlinfo
df6 = df6['querier'].groupby(df6['report_id']).apply(len).reset_index()
df6.to_csv('clean_data_cu/recordsmr.csv')
#==================================================
'''

contest_basic_test = pd.read_table('contest_basic_test.tsv')
#10000,10
contest_basic_train = pd.read_table('contest_basic_train.tsv')
#30000,11

le = LabelEncoder()
contest_basic_train['IS_LOCAL'] = le.fit_transform(contest_basic_train['IS_LOCAL'])
#0:native 1:nonnative

#0:native 1:nonnative
contest_basic_train['MARRY_STATUS'] = le.fit_transform(contest_basic_train['MARRY_STATUS'])

edu_to_num = {
     '专科':4,
     '专科及以下':3,
     '其他':0,
     '初中':1,
     '博士研究生':8,
     '本科':5,
     '硕士及以上':7,
     '硕士研究生':6,
     '高中':2
 }
contest_basic_train['EDU_LEVEL'] = contest_basic_train['EDU_LEVEL'].map(edu_to_num)

print(contest_basic_train.head(2))
'''







