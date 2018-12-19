# -*- coding: utf-8 -*-

class People:
    name = 'zhouhang'
    appear = 'handsome'

    def __init__(self):
        pass

p1 = People()
p1.name = 'alading'
People.appear = 'just so so'
print(p1.name,p1.appear)
print(People.name,People.appear)