#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: jd.py 
@time: 2018/03/{DAY} 
"""

import requests
import time
import random
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json

headers = {
'user-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36',
'accept':'*/*',
'accept-language':'zh-CN,zh;q=0.9',
'referer':'https://item.jd.com/5089253.html'
}

pro = ['197.161.74.17','197.161.74.5','50.233.137.36','124.128.39.138','113.200.56.13','142.44.137.222',
       '50.233.137.39','50.233.137.34','50.233.137.33','218.26.217.77','218.107.137.197','61.4.184.180',
       '59.67.152.230','114.112.104.223','50.233.137.37','50.28.48.83','203.6.149.156','186.4.227.159']

#https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv41880&productId=5089253&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1
#设置URL的第一部分
url1='https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv41880&productId=5089253&score=0&sortType=5&page='
#设置URL的第二部分
url2='&pageSize=10&isShadowSku=0&fold=1'
#乱序输出0-80的唯一随机数

def crawl_jd(url,pro):
    data = []
    res = requests.get(url,headers=headers,proxies={'http': random.choice(pro)}).text
    result = res.lstrip('fetchJSON_comment98vv41880(').rstrip(');')
    js_result = json.loads(result)['comments']
    for each in js_result:
        data_dict = {
            'content':each['content'],
            'creationTime':each['creationTime']}
        data.append(data_dict)
    return data

#300 ,300 的爬
datas = []
for i in range(69):
    url = url1+str(i+1)+url2
    data = crawl_jd(url,pro)
    datas.extend(data)
    # time.sleep(1.0)

df_datas = pd.DataFrame(datas,columns=['creationTime','content'])
df_datas.to_csv("comments.csv")





















