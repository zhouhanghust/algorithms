# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("szcfzs/sczs.csv",header=0)
data = np.array(df['spzs'])
normalize_data=(data-np.mean(data))/np.std(data)

df['spzs'] = normalize_data

df.to_csv("normalized_data.csv")



