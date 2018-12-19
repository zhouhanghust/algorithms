#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:27:24 2017

@author: caiyunxin
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import selenium.webdriver.support.ui as ui
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.PhantomJS('/Users/caiyunxin/Documents/phantomjs')  
url='https://m.weibo.cn/p/1005051866405545'
driver.get(url) 
#wait =ui.WebDriverWait(driver, 10)
pageSource = driver.page_source
soup=BeautifulSoup(pageSource,"html.parser")
##  //div[@class='card m-panel card9 weibo-member']//div[@class='weibo-og']/a//text()
## //div[@class='card m-panel card9 weibo-member']//div[@class='weibo-text']/text()
##  //div[@class='card m-panel card9 weibo-member']//div[@class='weibo-text']//span//text()
## //div[@class='weibo-media-wraps']//h3
wb_rumorlist=[]
div0_list=soup.find_all('div',{"class":"card m-panel card9 weibo-member"})
for div in div0_list:  
    rumor=''
    div11=soup.find('div',{"class":"weibo-og"})
    a=div11.find('a')
    rumor+=a.text
    div12=soup.find('div',{"class":"weibo-text"})
    rumor+=div12.text
    span_list=div.find_all('span')
    for span in span_list:
        rumor+=span.text
    div13=soup.find('div',{"class":"weibo-media-wraps"})
    h3=div13.find('h3')
    rumor+='\n'+h3.text
    wb_rumorlist.append(rumor)
    
    