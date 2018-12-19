#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: crawl_TB_demo.py 
@time: 2018/03/{DAY} 
"""

import re
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

dcap = dict(DesiredCapabilities.CHROME)
dcap["phantomjs.page.settings.userAgent"] = (
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0 ")

browser = webdriver.Chrome(desired_capabilities=dcap)
wait = WebDriverWait(browser,10)

def click_PL():
    try:
        browser.get('https://www.tmall.com')
        browser.maximize_window()
        input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#mq')))
        enter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#mallSearch > form > fieldset > div > button')))
        input.send_keys('iphoneX手机壳')
        enter.click()
        enter_item = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#J_ItemList > div:nth-child(5) > div')))
        enter_item.click()
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#J_TabBar > li.tm-selected > a')))
        submit.click()
    except TimeoutException:
        return click_PL()

def next_page(pageNumber):
    try:
        time.sleep(0.2)
        browser.execute_script("window.scrollTo(0,2500)")
        # time.sleep(2)
        # write = browser.find_element_by_css_selector('#comment-0 > div.com-table-footer > div > div > a.ui-pager-next')
        # ActionChains(browser).move_to_element(write).perform()
        time.sleep(1)
        submit = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#J_Reviews > div > div.rate-page > div > a:nth-child(7)')))
        submit.click()
        wait.until(EC.text_to_be_present_in_element(
            (By.CSS_SELECTOR, '#J_Reviews > div > div.rate-page > div > span:nth-child(3)'), str(pageNumber)))
    except TimeoutException:
        return next_page(pageNumber)


click_PL()

# for i in range(2,1000):
#     next_page(i)