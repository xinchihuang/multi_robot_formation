import cv2
import numpy as np

path = "/training_data/data/6/occupancy_maps.npy"
maps = np.load(path)
print(maps.shape)
print(maps[0])
for i in range(maps.shape[0]):
    cv2.imshow("maps", maps[i])
    cv2.waitKey(0)
