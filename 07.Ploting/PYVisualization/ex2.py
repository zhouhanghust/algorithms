# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

N = 7
winnersplot = (142.6,125.3,62.0,81.0,145.6,319.4,178.1)
ind = np.arange(N)

width = 0.35

fig,ax = plt.subplots()
winners = ax.bar(ind,winnersplot,width,color='#ffad00')

nomineesplot = (109.4,94.8,60.7,44.6,116.9,262.5,102.0)
nominees = ax.bar(ind+width,nomineesplot,width,color='#9b3c38')

ax.set_xticks(ind+width)
ax.set_xticklabels(('Best Picture','Director','Best Actor','Best Actress','Editing','Visual Effects','Cinematography'))
ax.legend((winners[0],nominees[0]),('Academy Award Winners','Academy Award Nominees'))
ax.grid()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        hcap = "$" + str(height) + "M"
        ax.text(rect.get_x()+rect.get_width()/2.,height,hcap,ha='center',va='bottom',rotation='vertical')

autolabel(winners)
autolabel(nominees)

print(nominees[0])

plt.show()