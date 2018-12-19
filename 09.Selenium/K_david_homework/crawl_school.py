# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

headers = {
'Host':'www.zuel.edu.cn',
'Upgrade-Insecure-Requests':'1',
'User-Agent':'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'
}
url = "http://www.zuel.edu.cn/"
res = requests.get(url,headers=headers)
res.encoding = 'utf-8'

soup =  BeautifulSoup(res.text,'lxml')

result = soup.select('select.w16_openLink')
# print(result)
option_list = result[0].select('option')
del option_list[:2]
academy = {}

for each_option in option_list:
    academy[each_option.text] = each_option['value']
#
# json_academy = json.dumps(academy)
# with open('academy_links.json','w') as f:
#     f.write(json_academy)




# news = soup.select('div.jianj a')
# first_new_link = url + news[0]['href']
# first_new_contents = news[0].text





# JZGG = soup.select('ul.news_list.zdy-6')
# # print(JZGG)
# jzcontents = JZGG[0].select('a')
# # print(jzcontents)
# jzlist = []
# for each_a in jzcontents:
#     data={
#         'link':each_a['href'],
#         'title':each_a['title'],
#     }
#     jzlist.append(data)
# print(pd.DataFrame(jzlist))


