# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy import  stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


y = np.load('mc_data.npy')
data = pd.Series(y)
diff1 = data.diff(1)
diff2 = data.diff(2)
# plt.figure()
# plt.plot(diff2[:500])
# plt.show()

dta = diff2
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=10,ax=ax2)
plt.show()



















































# df = pd.read_csv("szcfzs/sczs.csv",header=0)
# print(df['time'][1500:])
# data = df['spzs'][1500:]
# plt.figure()
# plt.plot(data)
# plt.show()















