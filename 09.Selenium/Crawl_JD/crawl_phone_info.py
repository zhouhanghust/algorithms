#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: crawl_phone_info.py 
@time: 2018/03/{DAY} 
"""

import re             #使用正则表达式提取数据
from selenium import webdriver         #用于打开浏览器
from selenium.common.exceptions import TimeoutException     #超时异常
from selenium.webdriver.common.by import By                 #用于定位器定位
from selenium.webdriver.support.ui import WebDriverWait     #用于设置等待时间和超时时间
from selenium.webdriver.support import expected_conditions as EC      #用于写条件判断
from bs4 import BeautifulSoup          #用于解析网页
import time
import pandas as pd


browser = webdriver.Chrome()               #实例化chrome浏览器类，可看作打开一个chrome浏览器
wait = WebDriverWait(browser,10)           #设置超时时间为10s

def search():                              #定义搜索函数
    try:                                   #try...except...用于捕捉异常
        browser.get('https://www.jd.com')     #打开京东页面
        browser.maximize_window()                 #全屏浏览器
        input = wait.until(                       #定位到搜索输入框
            EC.presence_of_element_located((By.CSS_SELECTOR,"#key"))
        )
        submit = wait.until(                      #定位到搜索框的搜索键
            EC.element_to_be_clickable((By.CSS_SELECTOR,'#search > div > div.form > button')))
        input.send_keys('手机')                    #输入需要查询的商品
        submit.click()                            #点击
        total = wait.until(                       #显示商品页面共多少页
            EC.presence_of_element_located((By.CSS_SELECTOR,'#J_bottomPage > span.p-skip > em:nth-child(1) > b')))
        return total.text                         #输出商品页面总页数
    except TimeoutException:                      #捕捉超时异常
        return search()                           #若超时，递归调用自身，再次尝试


def next_page(pageNumber):                        #定义翻页函数
    try:
        browser.execute_script("window.scrollTo(0,9000)")      #滑动浏览器滚动条至翻页栏
        time.sleep(2)
        input = wait.until(                       #定位到输入需转到的页面输入框
            EC.presence_of_element_located((By.CSS_SELECTOR, "#J_bottomPage > span.p-skip > input"))
        )
        submit = wait.until(                      #定位到翻页确认键
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#J_bottomPage > span.p-skip > a')))
        input.clear()                             #清空页面输入框
        input.send_keys(pageNumber)               #输入需要跳转的页面
        submit.click()                            #点击翻页确认键
        wait.until(                               #检查是否跳转到需要的页面
            EC.text_to_be_present_in_element((By.CSS_SELECTOR,'#J_bottomPage > span.p-num > a.curr'),str(pageNumber)))
        return get_products()
    except TimeoutException:                      #捕捉超时异常
        next_page(pageNumber)                     #若超时，递归调用自身，再次尝试

p = re.compile('<[^>]+>')                         #使用正则表达式去除html标签

def get_products():                               #定义抓取页面信息函数
    browser.execute_script("window.scrollTo(0,6000)")     #滑动浏览器滚动条至翻页栏
    time.sleep(2)                                 #等待2s，为演示浏览器已滚动至翻页栏
    wait.until(                                   #等待页面商品信息加载完成
        EC.presence_of_element_located((By.CSS_SELECTOR,'#J_goodsList li.gl-item')))
    html = browser.page_source                    #取出当前页面的html
    soup = BeautifulSoup(html,'lxml')             #用BeautifulSoup解析网页
    items = soup.select('#J_goodsList li.gl-item')   #提取所需要的信息
    result = []                                   #定义空列表用于存放提取的信息
    for item in items:
        product = {}                              #在循环内部定义字典用于存放提取的信息
        try:                                      #观察后知道图片链接提取方式不唯一，使用try...except...来解决该问题
            product['image'] = 'https:' + item.select('.p-img a img')[0]['src']
        except:
            product['image'] = 'https:' + item.select('.p-img a img')[0]['data-lazy-img']
        try:
            product['shop'] = item.select('.p-shop a')[0].text      #提取店家名称，有些商品店家名称为空，故用try...except...来解决
        except:
            product['shop'] = None
        product['price'] = item.select('.p-price strong i')[0].text #提取商品价格
        temp = str(item.select('.p-name.p-name-type-2 em')[0])      #为使用正则表达式需将html转成字符串
        product['item_name'] = str(p.sub("",temp)).strip()          #正则表达式提取商品名称
        result.append(product)                                      #将提取的信息存入列表
    return result                                                   #返回全部信息


total_pages = int(search())                                         #获得商品总页数
data = get_products()                                               #获得第一页的商品信息
for i in range(2,6):      #循环的末尾设置为 total_pages + 1 即可爬取所有页面
    temp = next_page(i)
    data.extend(temp)                                               #将其他页面的商品信息追加到data列表中

df = pd.DataFrame(data,columns=['item_name','price','shop','image'])     #将列表转成数据框
print(len(df))                                                      #打印数据框行数
print(df.head(5))                                                   #打印数据框前5条数据

# df.to_csv("phones.csv")                                             #将数据保存为csv格式











