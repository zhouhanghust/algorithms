# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import threading
import os

with open('academy_links.json','r') as f:
    academy = f.read()

academy = json.loads(academy)
#print(academy)


class myThread (threading.Thread):
    def __init__(self, name, url, headers):
        threading.Thread.__init__(self)
        self.name = name
        self.url = url
        self.headers = headers
    def run(self):
        print ("开始线程：" + self.name)
        #threadLock.acquire()
        self.soup,self.jianjie_url = crawl(self.url,self.headers)
        if self.jianjie_url.endswith('m'):
            self.introduction = crawl_introduction(self.jianjie_url,self.headers)
        else :
            self.introduction = ''
        if self.introduction :
            filename = '{}.txt'.format(self.name)
            if os.path.isfile(filename):
                print('File {0} already exists. Deleting file...'.format(filename))
                os.remove(filename)
            f=open(filename, mode='x',encoding='utf-8')
            f.write(self.name+'\n')
            f.write(self.introduction)
            f.close()
        #threadLock.release()
        print ("退出线程：" + self.name)

#threadLock = threading.Lock()

def crawl(url, headers):
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'lxml')
    a = soup.find('a', attrs={'title': '学院简介'})
    if a:
        jianjie_url = url + a['href']
    else :
        jianjie_url = url
    return soup,jianjie_url
    
    
    
def crawl_introduction(url,headers):
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'lxml')
    div=soup.find('div', attrs={'class': 'paging_content'})
    introduction=''
    if div:
        p_list=div.find_all('p')
        for p in p_list:
            introduction=introduction+p.text
    return introduction   
    
    
names = list(academy.keys())
# 创建新线程
threads = []
for each_name in names:
    each_url = academy[each_name]
    each_headers = {
        'Host':'%s' %each_url[7:-1],
        're-Requests':'1',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'
        }
    thread = myThread(each_name, each_url, each_headers)
    threads.append(thread)


for each_thread in threads:
    each_thread.start()

for each_thread in threads:
    each_thread.join()
