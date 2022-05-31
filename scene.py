"""
A scene template
author: Xinchi Huang
"""
import vrep_interface
from robot import robot
import math
import random


class scene:
    def __init__(self, num_robot):
        self.num_robots = num_robot
        self.robot_list = []
        self.adjacency_matrix = []
        self.client_id = None

    def initial_vrep(self):
        self.client_id = vrep_interface.init_vrep()
        return self.client_id

    def add_robot(self, robot_index):
        new_robot = robot()
        (
            robot_handle,
            motor_left_handle,
            motor_right_handle,
            point_cloud_handle,
        ) = vrep_interface.get_vrep_handle(self.client_id, robot_index)

        new_robot.executor.client_id = self.client_id
        new_robot.executor.robot_handle = robot_handle
        new_robot.executor.motor_left_handle = motor_left_handle
        new_robot.executor.motor_right_handle = motor_right_handle
        new_robot.executor.point_cloud_handle = point_cloud_handle

        new_robot.sensor.client_id = self.client_id
        new_robot.sensor.robot_index = robot_index
        new_robot.sensor.robot_handle = robot_handle

        self.robot_list.append(new_robot)

    def update_adjacency_matrix(self):
        return None

    def set_one_robot_pose(self, robot_handle, position, orientation):
        vrep_interface.post_robot_pose(
            self.client_id, robot_handle, position, orientation
        )

    def reset_pose(self, max_disp_range, min_disp_range):
        """
        Reset all robot poses in a circle
        :param max_disp_range: min distribute range
        :param min_disp_range: max distribute range


        pose_list:[[x,y,theta],[x,y,theta]]
        z0: A default parameter for specific robot and simulator. Make sure the robot is not stuck in the ground
        """
        pose_list = []
        for i in range(self.num_robots):
            while True:
                alpha = math.pi * (2 * random.random())
                rho = max_disp_range * random.random()
                x = rho * math.cos(alpha)
                y = rho * math.sin(alpha)
                theta = 2 * math.pi * random.random()
                too_close = False
                for p in pose_list:
                    if (x - p[0]) ** 2 + (y - p[1]) ** 2 <= min_disp_range**2:
                        too_close = True
                        break
                if too_close:
                    continue
                pose_list.append([x, y, theta])
                break
        for i in range(self.num_robots):
            z0 = 0.1587
            position = [pose_list[i][0], pose_list[i][1], z0]
            orientation = [0, 0, pose_list[i][2]]
            robot_handle = self.robot_list[i].executor.robot_handle
            vrep_interface.post_robot_pose(
                self.client_id, robot_handle, position, orientation
            )
