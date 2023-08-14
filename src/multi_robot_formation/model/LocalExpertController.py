import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)

from comm_data import ControlData
import numpy as np
import math
from ..utils.gabreil_graph import get_gabreil_graph_local

class LocalExpertControllerOld:
    def __init__(self,desired_distance=2,safe_margin=0.5):
        self.desired_distance = desired_distance
        self.name="LocalExpertController"
        self.safe_margin=safe_margin
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
        for i in range(len(position_array)):
            # print(neighbor)
            if neighbor[i]==1:
                num_neighbors+=1
                if position_array[i][0]==float("inf") or position_array[i][1]==float("inf"):
                    continue
                distance = (position_array[i][0]** 2 + position_array[i][1]** 2)**0.5
                # print(position_array[i])
                # print(distance)
                rate = ((distance) - self.desired_distance) / (distance-self.safe_margin)
                velocity_x = rate * (-position_array[i][0])
                velocity_y = rate * (-position_array[i][1])
                velocity_sum_x -= velocity_x
                velocity_sum_y -= velocity_y
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y

        return out_put

class LocalExpertControllerPartial:
    def __init__(self,desired_distance=2,safe_margin=0.5,view_range=5,view_angle=120):
        self.desired_distance = desired_distance
        self.name="LocalExpertControllerPartial"
        self.safe_margin=safe_margin
        self.view_range=view_range
        self.view_angle=view_angle
    def get_control(self,position_list_local):
        """
        :param position_list_local: local position list for training
        """

        position_array=np.array(position_list_local)
        out_put = ControlData()
        neighbor=np.ones(len(position_list_local))

        for v in range(len(position_list_local)):
            if np.linalg.norm(position_array[v])>self.view_range:
                neighbor[v]=0
            if abs(math.atan2(position_array[v][1],position_array[v][0]))>self.view_angle/2:
                neighbor[v] = 0
            m = (position_array[v]) / 2
            for w in range(len(position_list_local)):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(m):
                    neighbor[v] = 0
        velocity_sum_x =0
        velocity_sum_y =0
        omerga_sum=0
        num_neighbors=0
        for i in range(len(position_array)):
            # print(neighbor)
            if neighbor[i]==1:
                num_neighbors+=1
                if position_array[i][0]==float("inf") or position_array[i][1]==float("inf"):
                    continue
                distance = (position_array[i][0]** 2 + position_array[i][1]** 2)**0.5
                # print(position_array[i])
                # print(distance)
                rate = ((distance) - self.desired_distance) / (distance-self.safe_margin)
                velocity_x = rate * (-position_array[i][0])
                velocity_y = rate * (-position_array[i][1])
                omega=-math.atan2(position_array[i][1],position_array[i][0])

                velocity_sum_x -= velocity_x
                velocity_sum_y -= velocity_y
                omerga_sum -= omega
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y
        out_put.omega = omerga_sum
        return out_put

class LocalExpertController:
    def __init__(self,desired_distance=2,safe_margin=0.5):
        self.desired_distance = desired_distance
        self.name="LocalExpertController"
        self.safe_margin=safe_margin
    def leading_angle(self,gamma1,gamma2):
        if gamma1>=gamma2:
            return gamma1 - gamma2
        else:
            return gamma1 - gamma2+math.pi*2
    def get_control(self,pose_list,robot_id,sensor_range,sensor_angle):
        """
        :param position_list: local position list for training
        """
        out_put = ControlData()
        desired_distance = self.desired_distance
        gabreil_graph_local = get_gabreil_graph_local(pose_list, sensor_range, sensor_angle)
        neighbor_list = gabreil_graph_local[robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        velocity_sum_omega = 0
        robot_orientation = pose_list[robot_id][2]
        for neighbor_id in range(len(neighbor_list)):
            if neighbor_id == robot_id or neighbor_list[neighbor_id] == 0:
                continue
            distance = ((pose_list[robot_id][0] - pose_list[neighbor_id][0]) ** 2 + (
                        pose_list[robot_id][1] - pose_list[neighbor_id][1]) ** 2) ** 0.5
            rate = (distance - desired_distance) / distance
            velocity_x = rate * (pose_list[robot_id][0] - pose_list[neighbor_id][0])
            velocity_y = rate * (pose_list[robot_id][1] - pose_list[neighbor_id][1])
            velocity_omega = robot_orientation - math.atan2((pose_list[neighbor_id][1] - pose_list[robot_id][1]),
                                                            (pose_list[neighbor_id][0] - pose_list[robot_id][0]))
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
            velocity_sum_omega -= velocity_omega

        velocity_sum_omega=0
        gamma_list=[]
        for neighbor_id in range(len(neighbor_list)):
            if neighbor_id == robot_id or neighbor_list[neighbor_id] == 0:
                continue
            gamma = -robot_orientation + math.atan2((pose_list[neighbor_id][1] - pose_list[robot_id][1]),
                                                            (pose_list[neighbor_id][0] - pose_list[robot_id][0]))
            gamma_list.append(gamma)

        if len(gamma_list)>=1:
            gamma_d=(max(gamma_list)+min(gamma_list))
            print(gamma_list,gamma_d)
            for gamma in gamma_list:
                gamma_L=gamma+sensor_angle/2
                gamma_R=gamma-sensor_angle/2
                omega=(gamma_d/self.leading_angle(0,gamma_R))+(gamma_d/self.leading_angle(gamma_L,0))
                print(gamma_d,omega,self.leading_angle(0, gamma_R), self.leading_angle(gamma_L, 0))
                velocity_sum_omega+=omega
            print(robot_id,velocity_sum_omega)

        vx = velocity_sum_x * math.cos(robot_orientation) + velocity_sum_y * math.sin(robot_orientation)
        vy = -velocity_sum_x * math.sin(robot_orientation) + velocity_sum_y * math.cos(robot_orientation)
        out_put.velocity_x=vx
        out_put.velocity_y=vy
        out_put.omega=velocity_sum_omega
        return out_put

