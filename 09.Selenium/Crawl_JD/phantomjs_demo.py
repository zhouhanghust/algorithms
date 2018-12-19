#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: phantomjs_demo.py 
@time: 2018/03/{DAY} 
"""
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

dcap = dict(DesiredCapabilities.PHANTOMJS)  # 设置userAgent
dcap["phantomjs.page.settings.userAgent"] = (
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:25.0) Gecko/20100101 Firefox/25.0 ")

obj = webdriver.PhantomJS(desired_capabilities=dcap)  # 加载网址
obj.get('http://www.baidu.com')  # 打开网址
obj.save_screenshot("1.png")  # 截图保存
obj.quit()  # 关闭浏览器。当出现异常时记得在任务浏览器中关闭PhantomJS，因为会有多个PhantomJS在运行状态，影响电脑性能

