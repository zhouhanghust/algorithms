#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:hang 
@file: SMOTE.py 
@time: 2018/02/{DAY} 
"""

import numpy as np
import pandas as pd

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1






df = pd.read_csv('afterGYH_train.csv',index_col=0)

dff = df[df['y']==1]
dff1 = dff.iloc[:,:-1]
dff2 = dff1.values

s=Smote(dff2,N=1400)
add_to_dff = s.over_sampling()

adf = pd.DataFrame(add_to_dff,columns=df.columns[:-1])
concatdf = pd.concat((df,adf))

from sklearn.utils import shuffle

concatdf_v = shuffle(concatdf)
concatdf_v.to_csv("expanded_train.csv")
