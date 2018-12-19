# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")\
mov = pd.read_csv('ucdavis.csv')

x = mov.sleep
y = mov.exercise
z = mov.gpa

cm = plt.cm.get_cmap('RdYlBu')
fig,ax = plt.subplots(figsieze=(12,10))

sc = ax.scatter(x,y,s=z*3,c=z,cmap=cm,linewidth=0.2,alpha=0.5)
ax.grid()
fig.colorbar(sc)

ax.set_xlabel('Production Cost',fontsize=14)
ax.set_ylabel('Gross Profits',fontsize=14)

plt.show()