#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: chrome_demo.py 
@time: 2018/03/{DAY} 
"""

import re
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

browser = webdriver.Chrome()
wait = WebDriverWait(browser,10)
def search():
    try:
        browser.get('https://www.taobao.com')
        input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR,"#q"))
        )
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#J_TSearchForm > div.search-button > button')))
        input.send_keys('美食')
        submit.click()
        total = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#mainsrp-pager > div > div > div > div.total')))
        return total.text
    except TimeoutException:
        return search()

def next_page(pageNumber):
    try:
        input = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#mainsrp-pager > div > div > div > div.form > input"))
        )
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mainsrp-pager > div > div > div > div.form > span.btn.J_Submit')))
        input.clear()
        input.send_keys(pageNumber)
        submit.click()
        wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR,'#mainsrp-pager > div > div > div > ul > li.item.active > span'),str(pageNumber)))
    except TimeoutException:
        next_page(pageNumber)

def get_products():
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#mainsrp-itemlist .items .item')))
    html = browser.page_source
    soup = BeautifulSoup(html,'lxml')
    items = soup.select('#mainsrp-itemlist .items .item')
    for item in items:
        product = {
            'image':'https:' + item.select('.pic a img')[0]["src"]

        }
        print(product['image'])

def main():
    total = search()
    total = int(re.compile('(\d+)').search(total).group(1))
    # for i in range(2,total+1):
    #     next_page(i)
    get_products()




if __name__ == '__main__':
    main()







