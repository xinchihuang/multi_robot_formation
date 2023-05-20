import cv2
import numpy as np
def preprocess(map):
    # map=cv2.dilate(map,np.ones((2,2)))
    map=cv2.blur(map,(5,5))
    _,map=cv2.threshold(map,253,255,0)
    return map