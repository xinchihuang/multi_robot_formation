"""
author: Xinchi Huang
"""
import cv2
import numpy as np
from  utils.preprocess import preprocess
import os
dir_path="/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_data_test"
for filename in os.listdir(dir_path):
    new_filename = str(int(filename)-9)
    os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))