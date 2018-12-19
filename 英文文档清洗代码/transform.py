# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:41:24 2017

@author: Administrator
"""
import json
import pandas as pd
import re
import numpy as np
import glob
import os


def clean_email_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text
    
    
 
def clean_split(filename):
    
    with open('{0}'.format(filename),'r') as f:
       result = f.read()
        
    data = json.loads(result)
    data = pd.DataFrame(data)   
        
        
    docs = data['contents']
    docs = docs.apply(lambda s: clean_email_text(s))
    
    data['contents'] = docs
    
    index = np.random.permutation(len(data))
    data = data.iloc[index,:]
    cutline = int(0.7 * len(data))
    train_data = data.iloc[:cutline,:]
    test_data = data.iloc[cutline:,:]
    
    return train_data,test_data

dir=glob.glob('*.json')


for filename in dir:
    train_data,test_data = clean_split(filename)
    json_result_train = train_data.to_json(orient='index')
    json_result_test = test_data.to_json(orient='index')
     
    filename_train = "{0}_{1}.json".format(filename[:-5],'train')
    filename_test = "{0}_{1}.json".format(filename[:-5],'test')
    
    if os.path.isfile(filename_train):
        print('File {0} already exists. Deleting file...'.format(filename_train))
        os.remove(filename_train)
    with open(filename_train,'w') as f_train:
        f_train.write(json_result_train)
        
    if os.path.isfile(filename_test):
        print('File {0} already exists. Deleting file...'.format(filename_test))
        os.remove(filename_test)
    with open(filename_test,'w') as f_test:
        f_test.write(json_result_test)













