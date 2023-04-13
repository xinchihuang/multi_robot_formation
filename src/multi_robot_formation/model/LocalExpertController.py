import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)

from comm_data import ControlData
import numpy as np

class LocalExpertController:
    def __init__(self,desired_distance=2):
        self.desired_distance = desired_distance
        self.name="CentralizedController"
    def get_control(self,position_list_local):
        """
        :param position_list_local: local position list for training
        """
        position_array=np.array(position_list_local)
        out_put = ControlData()
        neighbor=np.ones(len(position_list_local))
        for v in range(len(position_list_local)):
            m = (position_array[v]) / 2
            for w in range(len(position_list_local)):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(m):
                    neighbor[v]=0
        velocity_sum_x =0
        velocity_sum_y =0
        num_neighbors=0
        for i in range(len(position_list_local)):
            if neighbor[i]==1:
                num_neighbors+=1
                if position_array[i][0]==float("inf") or position_array[i][1]==float("inf"):
                    continue
                distance = (position_array[i][0]** 2 + position_array[i][1]** 2)**0.5
                rate = (distance - self.desired_distance) / distance
                velocity_x = rate * (-position_array[i][0])
                velocity_y = rate * (-position_array[i][1])
                velocity_sum_x -= velocity_x
                velocity_sum_y -= velocity_y
        out_put.velocity_x = velocity_sum_x/num_neighbors
        out_put.velocity_y = velocity_sum_y/num_neighbors

        return out_put