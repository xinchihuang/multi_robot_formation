"""
author: Xinchi Huang
"""
from squaternion import Quaternion
q = Quaternion(0,0,1,1)
print(q.to_euler(degrees=False))