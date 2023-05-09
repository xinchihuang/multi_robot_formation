"""
author: Xinchi Huang
"""
import cv2
import numpy as np
image1=cv2.imread("0.jpg")
image2=cv2.imread("0s.jpg")
cv2.imshow("before", np.concatenate([image1,image2],axis=1))
image1=cv2.dilate(image1,np.ones((2,2)),iterations=1)
image2=cv2.dilate(image2,np.ones((2,2)),iterations=1)
cv2.imshow("dilate", np.concatenate([image1,image2],axis=1))
image1=cv2.blur(image1,(5,5))
image2=cv2.blur(image2,(5,5))
cv2.imshow("blur", np.concatenate([image1,image2],axis=1))

_,image1=cv2.threshold(image1,250,255,0)
_,image2=cv2.threshold(image2,250,255,0)
cv2.imshow("after", np.concatenate([image1,image2],axis=1))
cv2.waitKey(0)

def preprocess(map):
    map=cv2.dilate(map,np.ones((2,2)))
    map=cv2.blur(map,(5,5))
    _,map=cv2.threshold(map,250,255,0)
    return map