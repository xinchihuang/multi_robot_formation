import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)
import numpy as np
import math
#
# from ..comm_data import ControlData
# from ..utils.gabreil_graph import get_gabreil_graph_local,global_to_local
from comm_data import ControlData
from utils.gabreil_graph import get_gabreil_graph_local,global_to_local

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


class LocalExpertController:
    def __init__(self,desired_distance=2,sensor_range=5,sensor_angle=math.pi/2,safe_margin=0.4,K_f=1,K_m=1,K_omega=1,max_speed=1,max_omega=1):
        self.name = "LocalExpertController"
        self.desired_distance = desired_distance
        self.sensor_range = sensor_range
        self.sensor_angle = sensor_angle
        self.safe_margin =safe_margin
        self.K_f = K_f
        self.K_m = K_m
        self.K_omega = K_omega
        self.max_speed=max_speed
        self.max_omega=max_omega
        # print(self.safe_margin)
    def get_control(self,robot_id,pose_list):
        """
        :param position_list: global position list for training
        """

        out_put = ControlData()
        desired_distance = self.desired_distance
        gabreil_graph_local = get_gabreil_graph_local(pose_list, self.sensor_range, self.sensor_angle)
        pose_array_local=global_to_local(pose_list)
        neighbor_list = gabreil_graph_local[robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        velocity_sum_omega = 0
        # print(pose_list,pose_array_local)
        # print(robot_id,neighbor_list,gabreil_graph_local)
        for neighbor_id in range(len(neighbor_list)):
            if neighbor_id == robot_id or neighbor_list[neighbor_id] == 0:
                continue
            # position_local = [
            #     math.cos(pose_list[robot_id][2]) * (pose_list[neighbor_id][0]-pose_list[robot_id][0]) + math.sin(
            #         pose_list[robot_id][2]) * (pose_list[neighbor_id][1]-pose_list[robot_id][1]),
            #     -math.sin(pose_list[robot_id][2]) * (pose_list[neighbor_id][0]-pose_list[robot_id][0]) + math.cos(
            #         pose_list[robot_id][2]) * (pose_list[neighbor_id][1]-pose_list[robot_id][1])]
            position_local=pose_array_local[robot_id][neighbor_id]

            distance_formation = (position_local[0] ** 2 + position_local[1] ** 2) ** 0.5
            rate_f = (distance_formation - desired_distance) / distance_formation
            velocity_x_f = rate_f * position_local[0]
            velocity_y_f = rate_f * position_local[1]

            velocity_omega = math.atan2(position_local[1],(position_local[0]))
            # print(robot_id,neighbor_id,position_local[1],position_local[0],velocity_omega)

            gamma = math.atan2(position_local[1], (position_local[0]))
            distance_left = position_local[0] * math.sin(self.sensor_angle / 2) + position_local[1] * math.cos(
                self.sensor_angle / 2)
            if distance_left > self.safe_margin:
                rate_l=0
            else:
                # print(robot_id,neighbor_id,distance_left)
                rate_l = (self.safe_margin - distance_left) / distance_left
            velocity_x_l = rate_l * position_local[0]*(-math.sin(self.sensor_angle/2))
            velocity_y_l = rate_l * position_local[1] * (-math.cos(self.sensor_angle / 2))

            distance_right = position_local[0] * math.sin(self.sensor_angle / 2) - position_local[1] * math.cos(
                self.sensor_angle / 2)
            if distance_right > self.safe_margin:
                rate_r=0
            else:
                # print(robot_id, neighbor_id, distance_right)
                rate_r = (self.safe_margin - distance_right) / distance_right
            velocity_x_r = rate_r * position_local[0] * (-math.sin(self.sensor_angle / 2))
            velocity_y_r = rate_r * position_local[1] * (math.cos(self.sensor_angle / 2))

            distance_sensor = self.sensor_range - ((position_local[0]) ** 2 + (position_local[1])**2 )** 0.5
            if distance_sensor > self.safe_margin:
                rate_s=0
            else:
                # print(robot_id, neighbor_id, distance_sensor)
                rate_s = (self.safe_margin - distance_sensor) / distance_sensor
            velocity_x_s = rate_s * position_local[0] * (math.cos(gamma))
            velocity_y_s = rate_s * position_local[1] * (math.sin(gamma))

            velocity_sum_x += self.K_f*velocity_x_f+self.K_m*(velocity_x_l+velocity_x_r+velocity_x_s)
            velocity_sum_y += self.K_f*velocity_y_f+self.K_m*(velocity_y_l+velocity_y_r+velocity_y_s)
            velocity_sum_omega += self.K_omega*velocity_omega
        # print(robot_id,velocity_x_f,velocity_x_l,velocity_x_r,velocity_x_s)
        out_put.velocity_x=velocity_sum_x if abs(velocity_sum_x)<self.max_speed else self.max_speed*abs(velocity_sum_x)/velocity_sum_x
        out_put.velocity_y=velocity_sum_y if abs(velocity_sum_y)<self.max_speed else self.max_speed*abs(velocity_sum_y)/velocity_sum_y
        out_put.omega=velocity_sum_omega if abs(velocity_sum_omega)<self.max_omega else self.max_omega*abs(velocity_sum_omega)/velocity_sum_omega
        return out_put

# pose_lists=[[3, 0, math.pi/4],
#                  # [-2, 2, -math.pi/4],
#                  # [2, 2, -3*math.pi/4],
#                  # [2, -2, 3*math.pi/4],
#                  [0, 0, 0]]
# controller=LocalExpertController()
# control_i = controller.get_control(1,pose_lists)
# velocity_x, velocity_y,omega = control_i.velocity_x, control_i.velocity_y,control_i.omega
# print([velocity_x, velocity_y,omega])
