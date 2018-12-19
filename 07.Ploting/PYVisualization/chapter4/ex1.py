# -*- coding: utf-8 -*-
import scipy as sp

def multiplyPoly():
    cubic1 = sp.poly1d([3,4,5,5])
    cubic2 = sp.poly1d([4,1,-3,3])

    print(cubic1)
    print(cubic2)

    print('-'*36)

    print(cubic1*cubic2)

multiplyPoly()