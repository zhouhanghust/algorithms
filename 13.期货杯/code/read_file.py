# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pymysql
import sys




# contest_basic_test = pd.read_table('contest_basic_test.tsv')
# contest_basic_train = pd.read_table('contest_basic_train.tsv')
# contest_ext_crd_cd_ln_spl = pd.read_table('contest_ext_crd_cd_ln_spl.tsv')
# contest_ext_crd_cd_lnd = pd.read_table('contest_ext_crd_cd_lnd.tsv')
# contest_ext_crd_cd_lnd_ovd = pd.read_csv('contest_ext_crd_cd_lnd_ovd.csv')
# contest_ext_crd_hd_report = pd.read_csv('contest_ext_crd_hd_report.csv')
# contest_ext_crd_is_creditcue = pd.read_csv('contest_ext_crd_is_creditcue.csv')
# contest_ext_crd_is_ovdsummary = pd.read_csv('contest_ext_crd_is_ovdsummary.csv')
# contest_ext_crd_is_sharedebt = pd.read_csv('contest_ext_crd_is_sharedebt.csv')
# contest_ext_crd_qr_recorddtlinfo = pd.read_table('contest_ext_crd_qr_recorddtlinfo.tsv')
# contest_ext_crd_qr_recordsmr = pd.read_table('contest_ext_crd_qr_recordsmr.tsv')
contest_fraud = pd.read_table('contest_fraud.tsv')

# print(contest_fraud.dtypes)
# print(contest_fraud.head(10))


contest_fraud = contest_fraud.fillna('NULL')


# dbconn = pymysql.connect(
#     host = "127.0.0.1",
#     database = "dongzheng",
#     user = "root",
#     password = "hangzhou",
#     port = 3306,
#     charset = "utf8"
# )

# try:
#     with dbconn.cursor() as cursor:
#         sql = "insert into contest_fraud values (%s,%s)"
#         for i in range(len(contest_fraud)):
#             print("inserting: "+str(i))
#             cursor.execute(sql%(contest_fraud.iloc[i,0],contest_fraud.iloc[i,1]))
#             dbconn.commit()
#
# finally:
#     dbconn.close()
#     print('finally')



