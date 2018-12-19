# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:20:18 2017

@author: Administrator
"""

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


headers = {
'Host':'www.zuel.edu.cn',
'Upgrade-Insecure-Requests':'1',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'
}
url = "http://www.zuel.edu.cn/about/"
res = requests.get(url,headers=headers)
res.encoding = 'utf-8'

soup =  BeautifulSoup(res.text,'lxml')
text = soup.select('#wp_content_w26_0')
jianjie = text[0].text.replace('\u3000\u3000','\n\t')[2:]
print(jianjie)