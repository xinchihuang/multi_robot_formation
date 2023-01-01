import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_global_pose_array(pose_array):
    pose_array=pose_array[:,:2]
    for i in range(len(pose_array)):
        plt.scatter(pose_array[i][0],pose_array[i][1])
        plt.annotate(str(i),xy=pose_array[i])
    plt.show()
    plt.legend()
    print(pose_array)

# path = "/training_data/data/6/occupancy_maps.npy"
# maps = np.load(path)
# print(maps.shape)
# print(maps[0])
# for i in range(maps.shape[0]):
#     cv2.imshow("maps", maps[i])
#     cv2.waitKey(0)
