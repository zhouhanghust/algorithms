#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:27:36 2017

@author: caiyunxin
"""

#access Sinaweibo by sinaweibopy
from weibo import APIClient
import webbrowser        #python内置的包

APP_KEY = '694771429'#注意替换这里为自己申请的App信息
APP_SECRET = '052b3770829ae9329cb0d2e21be66645'
CALLBACK_URL = 'http://api.weibo.com/oauth2/default.html'#回调授权页面

#利用官方微博SDK
client = APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
#得到授权页面的url，利用webbrowser打开这个url
url = client.get_authorize_url()
print url
webbrowser.open_new(url)

#获取code=后面的内容
print '输入url中code后面的内容后按回车键：'
code = raw_input()
#code = your.web.framework.request.get('code')
#client = APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
r = client.request_access_token(code)
access_token = r.access_token # 新浪返回的token，类似abc123xyz456
expires_in = r.expires_in

# 设置得到的access_token
client.set_access_token(access_token, expires_in)

#可以打印下看看里面都有什么东西
#statuses = client.statuses__friends_timeline(count=100)['statuses'] #获取当前登录用户以及所关注用户（已授权）的微博</span>
#statuses+=client.statuses__friends_timeline(count=100,page=2)['statuses'] #获取当前登录用户以及所关注用户（已授权）的微博</span>
#statuses+=client.statuses__friends_timeline(count=100,page=3)['statuses'] #获取当前登录用户以及所关注用户（已授权）的微博</span>
#statuses+=client.statuses__friends_timeline(count=100,page=4)['statuses'] #获取当前登录用户以及所关注用户（已授权）的微博</span>
statuses = client.statuses__home_timeline(count=100)['statuses'] #获取当前登录用户以及所关注用户（已授权）的微博</span>
statuses+=client.statuses__home_timeline(count=100,page=2)['statuses']

length = len(statuses)
print length
#输出了部分信息
#username=[]
for i in range(0,length):
    #username.append(statuses[i]['user']['screen_name'])
    print u'昵称：'+statuses[i]['user']['screen_name']
    print u'简介：'+statuses[i]['user']['description']
    print u'位置：'+statuses[i]['user']['location']
    print u'微博：'+statuses[i]['text']

'''    
##################################
#可以把userName改成uid
def getUserAllComments(client, userName):
    i = 1
    comments = ""
    while (True):
        pieceComment = client.get.statuses__user_timeline(count=100,screen_name=userName, page=i)
            #已经获取到最早的微博信息，此时api返回的内容是空，所以以此为结束标记
        if (len(pieceComment["statuses"]) == 0):
            break
        i += 1 
            #控制输出内容格式和编码--直接显示中文，否则看到的是unicode编码
        content = json.dumps(pieceComment, ensure_ascii=False, indent=4, encoding="utf-8")
        comments += content
            
    return comments
    
uName = "微博辟谣"  
print getUserAllComments(client,uName) 


#######################
statuses0 = client.statuses__home_timeline()
import json
import re
str_sta = json.dumps(statuses0)
text_list = re.findall(r"(?<=\"text\": \").*?(?=\",)", str_sta)
for text in text_list:
    weibo = eval("u"+"'"+text+"'") 
    print weibo
    
##########
'''