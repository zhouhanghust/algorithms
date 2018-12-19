# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
contest_ext_crd_hd_report = pd.read_csv('contest_ext_crd_hd_report.csv')
# contest_ext_crd_hd_report['QUERY_REASON'] = le.fit_transform(contest_ext_crd_hd_report['QUERY_REASON'].values)
# print(contest_ext_crd_hd_report.head(10))
# print(np.unique(contest_ext_crd_hd_report['QUERY_REASON'].values))
# contest_ext_crd_hd_report[['REPORT_CREATE_TIME','QUERY_ORG']] = contest_ext_crd_hd_report[['REPORT_CREATE_TIME','QUERY_ORG']].astype(str)
# print(contest_ext_crd_hd_report[['REPORT_CREATE_TIME']].values)
contest_ext_crd_hd_report[['REPORT_CREATE_TIME']] = 'Null'
# print(contest_ext_crd_hd_report)
# contest_ext_crd_hd_report['QUERY_ORG'] = le.fit_transform(contest_ext_crd_hd_report['QUERY_ORG'].values)
# print(np.unique(contest_ext_crd_hd_report['QUERY_ORG'].values))
# print(contest_ext_crd_hd_report[200:500])

print(len(np.unique(contest_ext_crd_hd_report['REPORT_ID'].values)))





# dbconn = pymysql.connect(
#     host = "127.0.0.1",
#     database = "dongzheng",
#     user = "root",
#     password = "hangzhou",
#     port = 3306,
#     charset = "utf8"
# )
#
# try:
#     with dbconn.cursor() as cursor:
#         sql = "insert into contest_ext_crd_hd_report values (%s,%s,%s,%s)"
#         for i in range(len(contest_ext_crd_hd_report)):
#             print("inserting: "+str(i))
#             cursor.execute(sql%(contest_ext_crd_hd_report.iloc[i,0],contest_ext_crd_hd_report.iloc[i,1],contest_ext_crd_hd_report.iloc[i,2],contest_ext_crd_hd_report.iloc[i,3]))
#             dbconn.commit()
#
# finally:
#     dbconn.close()
#     print('finally')

