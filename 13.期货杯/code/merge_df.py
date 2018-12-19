# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import re
import time

os.chdir('D:\迅雷下载\东证期货杯-材料等\code')

contest_basic_test = pd.read_table('contest_basic_test.tsv')
#10000,10
contest_basic_train = pd.read_table('contest_basic_train.tsv')
#30000,11



contest_ext_crd_cd_ln = pd.read_table('contest_ext_crd_cd_ln.tsv')
#357196,22
df1 = contest_ext_crd_cd_ln
index = df1['class5_state'].notnull()
df1 = df1[index]
np.unique(df1['class5_state'])
mapdict = {
"次级":1,
"可疑":1,
"未知":0,
"关注":0,
"正常":0}
temp = df1['class5_state'].map(mapdict)
del df1['class5_state']
pp = ['loan_id','curr_overdue_cyc','curr_overdue_amount','recent_pay_date','scheduled_payment_date','open_date','end_date','remain_payment_cyc','payment_cyc','finance_org','type_dw','currency','guarantee_type','payment_rating']
for each in pp:
    del df1[each]
    
df1['class5_state'] = temp
mapdict2 = {'正常':0,'逾期':1}
df1['state'] = df1['state'].map(mapdict2)

def payment_state_ratio(lines):
    numsum = 0
    total = 0
    for line in lines:
        line = line.lstrip('/')
        numstr = re.sub("\D", "", line)
        total += len(line)-len(numstr)
        for each in numstr:
            numsum += int(each)
            total += int(each)
    return numsum/total
    

a1 = df1['payment_state'].groupby(df1['report_id'])
ratio24 = a1.apply(payment_state_ratio)

a2 = df1['scheduled_payment_amount']>df1['actual_payment_amount']
last_overdue = a2.map(int).groupby(df1['report_id']).max()

a3 = df1[['credit_limit_amount','balance']].groupby(df1['report_id']).sum()
debt_ratio = a3['balance']/a3['credit_limit_amount']

debt_ratio = debt_ratio.reset_index()
last_overdue = last_overdue.reset_index()
ratio24 = ratio24.reset_index()
temp_merge1 = pd.merge(df1,debt_ratio,on='report_id')
temp_merge1 = pd.merge(temp_merge1,last_overdue,on='report_id')
temp_merge1 = pd.merge(temp_merge1,ratio24,on='report_id')
temp_merge1.columns
temp_merge1.columns = ['report_id', 'dk_state', 'dk_class5_state', 'dk_debt_ratio', 'dk_last_overdue', 'payment_state']
temp_merge1.rename(columns={'payment_state':'dk_payment_state'})
temp_merge1.rename(columns={'payment_state':'dk_payment_state'},inplace=True)
temp_merge1.rename(columns={'dk_payment_state':'dk_ratio24'},inplace=True)

# ===================================================================


#contest_ext_crd_cd_ln_spl = pd.read_table('contest_ext_crd_cd_ln_spl.tsv')
#67725,6
contest_ext_crd_cd_lnd = pd.read_table('contest_ext_crd_cd_lnd.tsv')
#324229,20
df2 = contest_ext_crd_cd_lnd
df2['report_id']=df2['report_id'].astype(int)


index = df2['payment_state'].notnull()
df2 = df2[index]

b1 = df2['payment_state'].groupby(df2['report_id'])
djk_ratio24 = b1.apply(payment_state_ratio)

mapdict = {'正常':0,'冻结':1,'止付':1}
df2['state'] = df2['state'].map(mapdict)


delname = ['loancard_id','finance_org','open_date','guarantee_type','scheduled_payment_date','recent_pay_date','curr_overdue_cyc','curr_overdue_amount','cardtype']
for each in delname:
    del df2[each]

mapdict = {'人民币':1.00, '加拿大元':5.142, '日元':0.058, '欧元':7.867, '港元':0.810, '澳大利亚元':5.120, '澳门元':0.787, '瑞士法郎':6.712, '美元':6.33, '英镑':9.046}
df2['currency'] = df2['currency'].map(mapdict)

multiplyname = ['credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount','used_highest_amount','scheduled_payment_amount','actual_payment_amount']
for each in multiplyname:
    df2[each] = df2[each] * df2['currency']

ind = df2['currency'] != 1
df2[ind]
x = df2[ind]
ind2 = x['used_credit_limit_amount'] != 0.0
len(x[['used_credit_limit_amount']][ind2]) #212




ind = df2['currency'] != 1.0
df2 = df2[-ind]



aa = df2['payment_state'].groupby(df2['report_id'])
df2_ratio24 = aa.apply(payment_state_ratio)




temp3 = df2.groupby('report_id')[['credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount']].sum()
highest = df2[['used_highest_amount']].groupby(df2['report_id']).max()


onemonth = df2['scheduled_payment_amount']>df2['actual_payment_amount']
last_overdue = onemonth.map(int).groupby(df2['report_id']).max()

temp3 = pd.merge(temp3,highest,on='report_id')



#df2.groupby('report_id')['credit_limit_amount','used_credit_limit_amount','latest6_month_used_avg_amount','used_highest_amount','curr_overdue_cyc','curr_overdue_amount'].sum().reset_index()





contest_ext_crd_cd_lnd_ovd = pd.read_csv('contest_ext_crd_cd_lnd_ovd.csv')
#199644,4
df3 = contest_ext_crd_cd_lnd_ovd
df3.groupby('REPORT_ID')['LAST_MONTHS','AMOUNT'].sum()
#contest_ext_crd_hd_report = pd.read_csv('contest_ext_crd_hd_report.csv')
#40000,4
contest_ext_crd_is_creditcue = pd.read_csv('contest_ext_crd_is_creditcue.csv')
#39970,11
dff = contest_ext_crd_is_creditcue



dff.rename(columns=lambda x:x.lower(),inplace=True)

delname = ['first_sl_open_month']

def time2num(timestr):
    temp = time.strptime(timestr,"%Y.%m")
    timeseconds = time.mktime(temp)
    return timeseconds

    
 ind1 = dff['first_loan_open_month']=='--'   
 dff['first_loan_open_month'][ind1] = dff[ind1]['first_loancard_open_month'].values

ind2 = dff['first_loancard_open_month']=='--'
dff['first_loancard_open_month'][ind2]=dff[ind2]['first_loan_open_month']   
    
 
temp = (dff[['first_loan_open_month']]=='--').join(dff[['first_loancard_open_month']]=='--')
temp.apply(np.all,axis=1)
sum(temp.apply(np.all,axis=1))
ind3 = temp.apply(np.all,axis=1)
dff[ind3]
dff[-ind3]

dff['first_open_time']=np.nan
dff['first_open_time'][-ind3] = dff[-ind3][['first_loancard_open_month','first_loan_open_month']].applymap(time2num).apply(min,axis=1)

def maxmin(x):
    a = max(x)
    b = min(x)
    result = []
    for each in x:
        temp = (each-b)/(a-b)
        result.append(temp)
    return result
    
dff['first_open_time'][-ind3]=(dff[['first_open_time']][-ind3].apply(maxmin,axis=0)).values
del dff['first_loan_open_month']
del dff['first_loancard_open_month']
dff.columns


    
    
    
    
    
    
    
    
    
    
    
    
    


contest_ext_crd_is_ovdsummary = pd.read_csv('contest_ext_crd_is_ovdsummary.csv')
#76212,6
df4 = contest_ext_crd_is_ovdsummary
df4.groupby('REPORT_ID')['COUNT_DW','MONTHS','HIGHEST_OA_PER_MON','MAX_DURATION'].max().reset_index()
contest_ext_crd_is_sharedebt = pd.read_csv('contest_ext_crd_is_sharedebt.csv')
#76246,11
df5 = contest_ext_crd_is_sharedebt
df5.groupby('REPORT_ID')['CREDIT_LIMIT','BALANCE','USED_CREDIT_LIMIT','LATEST_6M_USED_AVG_AMOUNT'].sum().reset_index()





#contest_ext_crd_qr_recorddtlinfo = pd.read_table('contest_ext_crd_qr_recorddtlinfo.tsv')
#654329,4
#contest_ext_crd_qr_recordsmr = pd.read_table('contest_ext_crd_qr_recordsmr.tsv')
#760,3
#contest_fraud = pd.read_table('contest_fraud.tsv')
#40000,2

# le = LabelEncoder()
# contest_basic_train['IS_LOCAL'] = le.fit_transform(contest_basic_train['IS_LOCAL'])
# #0:native 1:nonnative
# contest_basic_train['MARRY_STATUS'] = le.fit_transform(contest_basic_train['MARRY_STATUS'])
# #
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
# contest_basic_train['EDU_LEVEL'] = contest_basic_train['EDU_LEVEL'].map(edu_to_num)
#
#
# table_list = [contest_ext_crd_cd_ln,contest_ext_crd_cd_ln_spl,contest_ext_crd_cd_lnd,contest_ext_crd_cd_lnd_ovd
#               ,contest_ext_crd_hd_report,contest_ext_crd_is_creditcue,contest_ext_crd_is_ovdsummary
#               ,contest_ext_crd_is_sharedebt,contest_ext_crd_qr_recorddtlinfo
#               ,contest_ext_crd_qr_recordsmr,contest_fraud]
#
# for each in table_list:
#     print(len(each))
#
#
#
#
#
#
# data = contest_basic_train
# sum(data.notnull().apply(np.all,axis=1))/len(data) #0.18049999999999999
#







