# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import LabelEncoder

contest_basic_train = pd.read_table('contest_basic_train.tsv')

le = LabelEncoder()
contest_basic_train['IS_LOCAL'] = le.fit_transform(contest_basic_train['IS_LOCAL'])
# contest_basic_train['EDU_LEVEL'] = le.fit_transform(contest_basic_train['EDU_LEVEL'])
contest_basic_train['MARRY_STATUS'] = le.fit_transform(contest_basic_train['MARRY_STATUS'])


edu_to_num = {
    "初中":0,
    "高中":1,
    "专科及以下":2,
    "专科":3,
    "本科":4
}

contest_basic_train['EDU_LEVEL'] = contest_basic_train['EDU_LEVEL'].map(edu_to_num)
contest_basic_train['WORK_PROVINCE'] = contest_basic_train['WORK_PROVINCE'].astype('int')
contest_basic_train = contest_basic_train.fillna('NULL')
print(contest_basic_train.dtypes)
# print(contest_basic_train.head(200))
print(contest_basic_train['WORK_PROVINCE'])


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
#         sql = "insert into contest_basic_train values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
#         for i in range(len(contest_basic_train)):
#             print("inserting: "+str(i))
#             cursor.execute(sql%(contest_basic_train.iloc[i,0],contest_basic_train.iloc[i,1],contest_basic_train.iloc[i,2],contest_basic_train.iloc[i,3]
#                                 , contest_basic_train.iloc[i,4],contest_basic_train.iloc[i,5],contest_basic_train.iloc[i,6]
#                                 , contest_basic_train.iloc[i,7],contest_basic_train.iloc[i,8],contest_basic_train.iloc[i,9]
#                                 , contest_basic_train.iloc[i,10]))
#             dbconn.commit()
#
# finally:
#     dbconn.close()
#     print('finally')


