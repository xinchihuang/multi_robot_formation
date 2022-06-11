"""
A scene template
author: Xinchi Huang
"""
import math
import random
from collections import defaultdict
import numpy as np
import vrep_interface

from robot import Robot
from utils import get_gabreil_graph


class Scene:
    """
    Scene for multiple robots
    """

    def __init__(self):
        """
        robot_list: A list contains all robot in the scene
        adjacency_list: A dict records robots' neighbor position and relative distance in gabreil graph
        {robot index:[(neighbor index, neighbor x, neighbor y,relative distance)..]..}
        client_id: A unique Id for the simulation environment
        """
        self.robot_list = []
        self.adjacency_list = defaultdict(list)
        self.client_id = None

    def initial_vrep(self):
        """
        initial Vrep get client id
        :return: A Verp client id
        """
        self.client_id = vrep_interface.init_vrep()
        return self.client_id

    def add_robot(self, robot_index):
        """
        Add a robot in the scene
        :param robot_index:
        :return:
        """
        new_robot = Robot()
        (
            robot_handle,
            motor_left_handle,
            motor_right_handle,
            point_cloud_handle,
        ) = vrep_interface.get_vrep_handle(self.client_id, robot_index)
        new_robot.index = robot_index

        new_robot.executor.client_id = self.client_id
        new_robot.executor.robot_handle = robot_handle
        new_robot.executor.motor_left_handle = motor_left_handle
        new_robot.executor.motor_right_handle = motor_right_handle
        new_robot.executor.point_cloud_handle = point_cloud_handle

        new_robot.sensor.client_id = self.client_id
        new_robot.sensor.robot_index = robot_index
        new_robot.sensor.robot_handle = robot_handle

        new_robot.sensor.get_sensor_data()
        self.robot_list.append(new_robot)

    def update_adjacency_list(self):
        """
        Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

        """
        # print("Distance")
        node_num = len(self.robot_list)
        # collect robots' position in th scene
        position_list = []
        index_list = []
        for i in range(node_num):
            index_list.append(self.robot_list[i].index)
            position = self.robot_list[i].sensor_data.position[:-1]
            position_list.append(position)
        position_array = np.array(position_list)

        # Get Gabreil Graph
        gabriel_graph = get_gabreil_graph(position_array, node_num)

        # Create adjacency list
        new_adj_list = defaultdict(list)
        for i in range(node_num):
            for j in range(node_num):
                if gabriel_graph[i][j] == 1 and not i == j:
                    distance = (
                        (position_array[i][0] - position_array[j][0]) ** 2
                        + (position_array[i][1] - position_array[j][1]) ** 2
                    ) ** 0.5
                    new_adj_list[index_list[i]].append(
                        (
                            index_list[j],
                            position_array[j][0],
                            position_array[j][1],
                            distance,
                        )
                    )
        self.adjacency_list = new_adj_list

        print("DISTANCE")
        for r in self.adjacency_list:
            for n in self.adjacency_list[r]:
                print("edge:",r,n[0],"distance:",n[3])



    def broadcast_adjacency_list(self):
        """
        Send adjacency list to all robots for centralized control
        :return:
        """
        for robot in self.robot_list:
            robot.network_data = self.adjacency_list

    def set_one_robot_pose(self, robot_handle, position, orientation):
        """

        :param robot_handle:
        :param position:
        :param orientation:
        :return:
        """
        vrep_interface.post_robot_pose(
            self.client_id, robot_handle, position, orientation
        )

    def reset_pose(self, max_disp_range, min_disp_range):
        """
        Reset all robot poses in a circle
        :param max_disp_range: min distribute range
        :param min_disp_range: max distribute range


        pose_list:[[pos_x,pos_y,theta],[pos_x,pos_y,theta]]
        height: A default parameter for specific robot and simulator.
        Make sure the robot is not stuck in the ground
        """
        pose_list = []
        for i in range(len(self.robot_list)):
            while True:
                alpha = math.pi * (2 * random.random())
                rho = max_disp_range * random.random()
                pos_x = rho * math.cos(alpha)
                pos_y = rho * math.sin(alpha)
                theta = 2 * math.pi * random.random()
                too_close = False
                for p in pose_list:
                    if (pos_x - p[0]) ** 2 + (pos_y - p[1]) ** 2 <= min_disp_range**2:
                        too_close = True
                        break
                if too_close:
                    continue
                pose_list.append([pos_x, pos_y, theta])
                break
        for i in range(len(self.robot_list)):
            pos_height = 0.1587
            position = [pose_list[i][0], pose_list[i][1], pos_height]
            orientation = [0, 0, pose_list[i][2]]
            robot_handle = self.robot_list[i].executor.robot_handle
            vrep_interface.post_robot_pose(
                self.client_id, robot_handle, position, orientation
            )
