# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure(figsize=(15,13))
plt.ylim(740,1128)
plt.xlim(1965,2011)
with open('mortality1.csv') as csvfile:
    mortdata = [row for row in csv.DictReader(csvfile)]

data = pd.DataFrame(mortdata)
x = list(data.Year.values)
males_y = list(data.Males.values)
females_y = list(data.Females.values)
every_y = list(data.Everyone.values)

plt.plot(x,males_y,color='blue',label='Males',linewidth=1.8)
plt.plot(x,females_y,color='red',label='Females',linewidth=1.8)
plt.plot(x,every_y,color='green',label='Everyone',linewidth=1.8)
plt.legend(loc=0,prop={'size':10})
plt.show()

