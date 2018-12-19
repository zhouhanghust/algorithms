# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(15,13))
plt.ylim(35,102)
plt.xlim(1965,2015)

colorsdata = ['red','blue','green','purple','lightblue','lightgreen','black']
labeldata = ['Below 25','25-44','45-54','55-64','65-74','75-84','over 85']

with open('mortality2.csv') as csvfile:
    mortdata = [row for row in csv.reader(csvfile)]

data = pd.DataFrame(mortdata)
data.columns = list('abcdefghj')
del data['j']

x = list(data.a)
y = []
for each in list('bcdefgh'):
    y.append(list(data[each]))

for col in range(0,7):
    if(col == 1):
        plt.plot(x,y[col],color=colorsdata[col],label=labeldata[col],linewidth=3.8)
    else :
        plt.plot(x,y[col],color=colorsdata[col],label=labeldata[col],linewidth=2)

plt.legend(loc=0,prop={'size':15})
plt.show()

