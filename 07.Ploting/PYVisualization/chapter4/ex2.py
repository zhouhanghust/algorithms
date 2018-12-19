# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps,romberg

a = -3.0
b = 3.0
N = 10

x = np.linspace(a,b,N)
y = 9 - x**2
yromb = lambda x:(9-x**2)

t = np.trapz(y,x)                   #三种方法分别积分
s = simps(y,x)
r = romberg(yromb,a,b)

aiv = (9*b - (b**3)/3.0) - (9*a - (a**3)/3.0)

print('trapezoidal = {0} ({1:%} error)'.format(t,(t-aiv)/aiv))
print('simpsons = {0} ({1:%} error)'.format(s,(s-aiv)/aiv))
print('romberg = {0} ({1:%} error)'.format(r,(r-aiv)/aiv))
print('actual value = {0}'.format(aiv))