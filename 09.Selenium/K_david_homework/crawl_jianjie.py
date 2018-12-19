# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import threading
import time
import os
import re

with open('academy_links.json', 'r') as f:
    academy = f.read()

academy = json.loads(academy)


# print(academy)

jianjie_dict = {
    '马克思主义学院': 'http://mkszyxy.zuel.edu.cn/1703/list.htm',
    '哲学院': 'http://zxy.zuel.edu.cn/2654/list.htm',
    '经济学院': 'http://jjxy.zuel.edu.cn/2747/list.htm',
    '财政税务学院': 'http://csxy.zuel.edu.cn/4929/list.htm',
    '金融学院': 'http://finance.zuel.edu.cn/1103/list.htm',
    '法学院': 'http://law.zuel.edu.cn/3709/list.htm',
    '刑事司法学院': 'http://cjs.zuel.edu.cn/2013/0730/c3111a65935/page.htm',
    '外国语学院': 'http://wgyxy.zuel.edu.cn/830/list.htm',
    '新闻与文化传播学院': 'http://xwcb.zuel.edu.cn/49/list.htm',
    '工商管理学院': 'http://gsxy.zuel.edu.cn/4111/list.htm',
    '会计学院(会硕中心)': 'http://kjxy.zuel.edu.cn/3916/list.htm',
    '公共管理学院(MPA中心)': 'http://ggglxy.zuel.edu.cn/959/list.htm',
    '统计与数学学院': 'http://tsxy.zuel.edu.cn/4806/list.htm',
    '信息与安全工程学院': 'http://xagx.zuel.edu.cn/2096/list.htm',
    '文澜学院': 'http://wls.zuel.edu.cn/3771/list.htm',
    '知识产权学院': 'http://www.iprcn.com/IL_Zxjs.aspx?News_PI=1',
    'MBA学院': 'http://mba.zuel.edu.cn/2504/list.htm',
    '继续教育学院(网络教育学院)': 'http://sce.zuel.edu.cn/2918/list.htm',
    '中韩新媒体学院':'http://zhxmt.zuel.edu.cn/2659/list.htm'
}


class myThread(threading.Thread):
    def __init__(self, name, url, headers):
        threading.Thread.__init__(self)
        self.name = name
        self.url = url
        self.headers = headers

    def run(self):
        print("开始线程：" + self.name)
        #threadLock.acquire()
        self.soup = crawl(self.url, self.headers)
        try:
            self.pattern = re.compile(r'.+(学院简介.+)',re.S)
            self.jianjie = self.pattern.match(self.soup.text).group(1)
        except:
            self.jianjie = self.soup.text
        filename = '{}.txt'.format(self.name)
        if os.path.isfile(filename):
            print('File {0} already exists. Deleting file...'.format(filename))
            os.remove(filename)
        f=open(filename, mode='x',encoding='utf-8')
        f.write(self.name+'\n')
        f.write(self.jianjie)
        f.close()
        #threadLock.release()
        print("退出线程：" + self.name)


#threadLock = threading.Lock()


def crawl(url, headers):
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'lxml')

    return soup


names = list(jianjie_dict.keys())
# 创建新线程
threads = []
for each_name in names:
    each_url = jianjie_dict[each_name]
    each_h = academy[each_name]
    each_headers = {
        'Host': '%s' % each_h[7:-1],
        're-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'
    }
    thread = myThread(each_name, each_url, each_headers)
    threads.append(thread)

for each_thread in threads:
    each_thread.start()

for each_thread in threads:
    each_thread.join()


# .replace('\xa0','')
#pattern = re.compile(r'.+(学院简介.+)%s'%threads[0].name,re.S)
#
#m = pattern.match(soup.text)