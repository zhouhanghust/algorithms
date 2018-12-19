#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: crawl_jd_demo.py 
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
        browser.get('https://item.jd.com/5089253.html')
        browser.maximize_window()
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#detail > div.tab-main.large > ul > li:nth-child(5)')))
        submit.click()
        # browser.execute_script("window.scrollTo(0,1000)")
        # submit = wait.until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, '#comment > div.mc > div.J-comments-list.comments-list.ETab > div.tab-main.small > ul > li:nth-child(4) > a')))
        # submit.click()

    except TimeoutException:
        return click_PL()

def next_page(pageNumber,position):
    try:
        time.sleep(0.2)
        browser.execute_script("window.scrollTo(0,%s)"%position)
        time.sleep(0.2)
        write = browser.find_element_by_css_selector('#comment-0 > div.com-table-footer > div > div > a.ui-pager-next')
        ActionChains(browser).move_to_element(write).perform()
        time.sleep(0.2)
        submit = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#comment-0 > div.com-table-footer > div > div > a.ui-pager-next')))
        submit.click()
        wait.until(EC.text_to_be_present_in_element(
            (By.CSS_SELECTOR, '#comment-0 > div.com-table-footer > div > div > a.ui-page-curr'), str(pageNumber)))
    except TimeoutException:
        position -= 500
        return next_page(pageNumber,position)


click_PL()

for i in range(2,1000):
    next_page(i,3000)





# write = browser.find_element_by_css_selector('#comment-0 > div.com-table-footer > div > div > a.ui-pager-next')
# ActionChains(browser).move_to_element(write).perform()

# browser.get('https://item.jd.com/5089253.html')
# browser.maximize_window()
#
# submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#detail > div.tab-main.large > ul > li:nth-child(5)')))
# submit.click()
# time.sleep(2)
#
# # js = "var q=document.documentElement.scrollTop=4000"
# # browser.execute_script(js)
# browser.execute_script("window.scrollTo(0,4000)")
#
