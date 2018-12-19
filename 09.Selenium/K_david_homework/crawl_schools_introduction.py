#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:33:15 2017

@author: caiyunxin
"""
#从学校首页找学院首页网址
#//div[@frag='窗口16']//option/@value

import requests as res
from bs4 import BeautifulSoup
#import time
#import json
#import re  
#import os.path  

def crawl_page(headers,url):   
    res1 = res.get(url,headers=headers)
    res1.encoding = 'utf-8'
    soup = BeautifulSoup(res1.text,'lxml')
    div = soup.find_all('div', attrs={'frag': '窗口16'})  
    schools_url = {}
    for each_div in div:
        option = each_div.select('option')
    for i in option:
        schools_url[i.text]=i['value']
    return schools_url

headers = {
    'Host':'www.zuel.edu.cn',
    'Referer':'https://www.baidu.com/link?url=sWLReo-pXB0jOlsgbTgCrWlv8YEApBbfjDZHTYVVv43iM1LHSTF3cvTLPJIiXJhd&wd=&eqid=b1eba1e50001ee28000000035a0a9878',
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
    'X-Requested-With':'XMLHttpRequest'
    }
url = 'http://www.zuel.edu.cn/'
schools_url=crawl_page(headers,url)

del schools_url['————————————']
del schools_url['学院导航']

#从学院首页找学院简介网址
#//div[@id='wp_nav_w1']//ul[@class='wp_nav']//li[@class='nav-item i2']//ul[@class='sub-nav']/li[@class='nav-item i2-1']/a/@href
#//ul[@class='sub-nav']/li[@class='nav-item i2-1']/a/@href

def crawl_introduction_url(headers,url):   
    res1 = res.get(url,headers=headers)
    res1.encoding = 'utf-8'
    soup = BeautifulSoup(res1.text,'lxml')
    a_list=soup.find('a', attrs={'title': '学院简介'})
    intro_url=''
    if a_list:
        intro_url=a_list['href']
    return intro_url

intro_url_list=[]
for each_url in schools_url.values():
    url=each_url
    headers = {
    'Host':'',
    'Referer':'https://www.baidu.com/link?url=sWLReo-pXB0jOlsgbTgCrWlv8YEApBbfjDZHTYVVv43iM1LHSTF3cvTLPJIiXJhd&wd=&eqid=b1eba1e50001ee28000000035a0a9878',
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
    'X-Requested-With':'XMLHttpRequest'
    }
    headers['Host']=url[7:-1]
    intro_url=crawl_introduction_url(headers,url)
    intro_url_list.append(url+intro_url)
    
#从学院简介页爬简介
#//div[@class='paging_content']/p/text()  

school_host_intro=list(zip(schools_url.keys(),schools_url.values(),intro_url_list))

def crawl_introduction(headers,url):   
    res1 = res.get(url,headers=headers)
    res1.encoding = 'utf-8'
    soup = BeautifulSoup(res1.text,'lxml')
    div=soup.find('div', attrs={'class': 'paging_content'})
    introduction=''
    if div:
        p_list=div.find_all('p')
        for p in p_list:
            introduction=introduction+p.text
    return introduction   

introduction={}
for index,each_url in enumerate(intro_url_list):
    url=each_url
    if url.endswith('m'):
        headers = {
                'Host':'',
                'Referer':'',
                'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
                'X-Requested-With':'XMLHttpRequest'
                }
        headers['Host']=(school_host_intro[index][1])[7:-1]
        headers['Referer']=school_host_intro[index][1]               
        introduction[(school_host_intro[index][0])]=crawl_introduction(headers,url)
        

        
        
        
        
        
        



    
    

