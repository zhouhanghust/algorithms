import os
import sys

# print(os.getcwd())
# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.path.abspath(os.path.dirname(os.getcwd())))
parpath = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(parpath)

filepath = os.path.join(parpath,'ooss/someone.txt')
print(filepath)
with open(filepath,"r") as f:
    data = f.readlines()

for each in data:
    print(each.strip())



