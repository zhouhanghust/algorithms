import sys
import os

o_path = os.path.abspath(os.path.dirname(os.getcwd()))
print(o_path)
sys.path.append(o_path)
print(sys.path)
import file1.os1


