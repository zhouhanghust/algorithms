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
        browser.get('https://detail.tmall.com/item.htm?spm=a220m.1000858.1000725.1.325f5a7c2bVd8R&id=560207352065&skuId=3703839467501&standard=1&user_id=268451883&cat_id=2&is_b=1&rn=f73abbab6a0af22af0cc4162fa549ddd')
        browser.maximize_window()
        write = browser.find_element_by_css_selector('#J_QRCodeLogin > div.login-links > a.forget-pwd.J_Quick2Static')
        ActionChains(browser).move_to_element(write).perform()

        change_dl = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#J_QRCodeLogin > div.login-links > a.forget-pwd.J_Quick2Static')))
        change_dl.click()

        input_ad = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#TPL_username_1')))
        input_code = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#TPL_password_1')))
        enter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR,'#J_SubmitStatic')))
        input_ad.send_keys('14527027068')
        input_code.send_keys('Zhang1991')
        enter.click()
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