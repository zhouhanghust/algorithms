import pandas as pd
import pymysql


db = pymysql.connect("localhost", "root", "root", "RUNOOB")
cursor = db.cursor()


def pyinsert(value):
    # SQL 插入语句
    sql = "INSERT INTO didi(driver_id, zqnyddds, zqnwcdds, zqnsjqxdds, " \
          "sjqxl, sjlx, sjxb, sjnl, sjrzsc, sjjl, clzj, clpl, " \
          "clnx, cljg, zjz, gpfzxxg, zwzxxg, lswd) " \
          "VALUES (%s, %s, %s, %s, %s, '%s', '%s', %s, %s, %s, %s, %s, %s, %s, %s, '%s', '%s', %s)"% (value[0] ,value[1] ,value[2] ,value[3] ,value[4] ,value[5] ,value[6] ,value[7] ,value[8] ,value[9] ,value[10] ,value[11] ,value[12] ,value[13] ,value[14] ,value[15] ,value[16] ,value[17])
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 执行sql语句
        db.commit()
        print("Insert complete!")
    except:
        # 发生错误时回滚
        db.rollback()
        print("Insert error!")



df = pd.read_csv("./cancel_sample_1.csv")
for i in range(len(df)):
    pyinsert(df.iloc[i,:].tolist())
    print("the %sth data completed!" % i)

# 关闭数据库连接
db.close()



