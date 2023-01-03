"""
Code for generating demo occupancy map from given points
author: Xinchi Huang
"""

import math
import numpy as np
import cv2

class MapSimulator:
    def __init__(self,robot_size=0.2,max_height=0.3,map_size=100,max_x=10,max_y=10,rotate=False):
        """
        :param robot_size: Size of robot in occupancy map
        :param max_height: points' horizontal range
        :param map_size: The size of occupancy map
        :param max_x: Max world x coordinate
        :param max_y: Max world y coordinate
        :param rotate: Control whether to rotate the occupancy map or not(True: local map, False: global map)
        """


        self.robot_size=robot_size
        self.max_height = max_height
        self.map_size = map_size
        self.max_x = max_x
        self.max_y = max_y
        self.rotate = rotate

    def arctan(self,x, y):
        if x == 0 and y > 0:
            theta = math.pi / 2
        elif x == 0 and y < 0:
            theta = math.pi * 3 / 2
        elif x == 0 and y == 0:
            theta = 0
        elif x > 0 and y == 0:
            theta = 0
        elif x < 0 and y == 0:
            theta = math.pi
        else:
            theta = math.atan(y / x)
            if x > 0 and y > 0:
                pass
            elif x < 0 and y > 0:
                theta = theta+math.pi
            elif x < 0 and y < 0:
                theta = theta-math.pi
            elif x > 0 and y < 0:
                theta = theta
        return theta

    def data_filter(self,world_point):
        """
        Filter out the points that out of sensor range
        :param world_point: Points in world coordinate
        :param max_x: points' max x coordinate (left/right)
        :param max_y: points' max y coordinate (depth/distance)
        :param max_height: points' horizontal range
        :param min_range: min distance between robots
        :return: Points within sensor range
        """

        x = world_point[0]
        y = world_point[1]
        z = world_point[2]
        min_range=2*self.robot_size
        if x > self.max_x or x < -self.max_x or y > self.max_y or y < -self.max_y or z < -self.max_height:  #
            return None
        if x < min_range and y < min_range and x > -min_range and y > -min_range:
            return None
        return [x, y, z]


    def rotation(self,world_point, self_orientation):
        """
        Rotate the points according to the robot orientation to transform other robot's position from global to local
        :param world_point: Other robot's positions
        :param self_orientation: Robot orientation
        :return:
        """
        x = world_point[0]
        y = world_point[1]
        z = world_point[2]
        theta = self_orientation
        x_relative = math.cos(theta) * x + math.sin(theta) * y
        y_relative = -math.sin(theta) * x + math.cos(theta) * y
        return [x_relative, y_relative, z]



    def blocking(self,position_lists_local):
        """
        Handle the blocking case, Remove the f
        :param position_lists_local:
        :return:
        """
        out_position_lists_local = []
        for self_i in range(len(position_lists_local)):
            position_lists_i = []
            for robot_j in range(len(position_lists_local[self_i])):
                x = position_lists_local[self_i][robot_j][0]
                y = position_lists_local[self_i][robot_j][1]
                theta = self.arctan(x, y)

                block = False
                for robot_k in range(len(position_lists_local[self_i])):
                    if robot_k == robot_j:
                        continue
                    x_k = position_lists_local[self_i][robot_k][0]
                    y_k = position_lists_local[self_i][robot_k][1]
                    if x**2 + y**2 < x_k**2 + y_k**2:
                        continue
                    x_k1 = x_k - (self.robot_size / 2) * math.sin(theta)
                    y_k1 = y_k + (self.robot_size / 2) * math.cos(theta)

                    x_k2 = x_k + (self.robot_size / 2) * math.sin(theta)
                    y_k2 = y_k - (self.robot_size / 2) * math.cos(theta)

                    theta_k_1 = self.arctan(x_k1, y_k1)
                    theta_k_2 = self.arctan(x_k2, y_k2)
                    if max(theta_k_1, theta_k_2) - min(theta_k_1, theta_k_2) < math.pi:
                        if theta_k_1 < theta < theta_k_2 or theta_k_2 < theta < theta_k_1:
                            block = True
                    else:
                        if theta > max(theta_k_1, theta_k_2) or theta < min(
                            theta_k_1, theta_k_2
                        ):
                            block = True
                if block == False:
                    position_lists_i.append(position_lists_local[self_i][robot_j])
            out_position_lists_local.append(position_lists_i)
        return out_position_lists_local


    def global_to_local(self,position_lists_global,self_orientation_global):
        """
        Get each robot's observation from global absolute position
        :param position_lists_global: Global absolute position of all robots in the world
        :return: A list of local observations
        """
        position_lists_local = []
        self_pose_list = []
        for i in range(len(position_lists_global)):
            x_self = position_lists_global[i][0]
            y_self = position_lists_global[i][1]
            z_self = position_lists_global[i][2]

            self_pose_list.append([x_self, y_self, z_self])
            position_list_local_i = []
            for j in range(len(position_lists_global)):
                if i == j:
                    continue
                point_local_raw=[position_lists_global[j][0] - x_self,position_lists_global[j][1] - y_self,position_lists_global[j][2] - z_self]
                if self.rotate:
                    point_local_rotated=self.rotation(point_local_raw,self_orientation_global[i])
                    point_local_raw=point_local_rotated
                point_local=self.data_filter(point_local_raw)

                if not point_local==None:
                    position_list_local_i.append(point_local)
            position_lists_local.append(position_list_local_i)
        position_lists_local = self.blocking(position_lists_local)
        return position_lists_local, self_pose_list




    def world_to_map(self,world_point, map_size, max_x, max_y):
        """
        Transform points from world coordinate to map coordinate
        :param world_point: points' world coordinate
        :param map_size: The size of occupancy map
        :param max_x: Max world x coordinate
        :param max_y: Max world y coordinate
        :return: points in map coordinate

        """
        if world_point == None:
            return None
        x_world = world_point[0]
        y_world = world_point[1]
        x_map = int((max_x - x_world) / (2 * max_x) * map_size)
        y_map = int((max_y - y_world) / (2 * max_y) * map_size)
        if 0 <= x_map < map_size and 0 <= y_map < map_size:
            return [x_map, y_map]
        return None


    def generate_map_one(self,
        position_list_local,
        robot_size,
        map_size,
        max_x,
        max_y,
        ):

        """
        Generate occupancy map
        :param position_list_local: All robots' map coordinate relative to the observer robot [x,y,z]
        :param self_orientation: Observer robots' orientation (float)
        :param robot_size: Size of robot in occupancy map
        :param max_height: points' horizontal range
        :param map_size: The size of occupancy map
        :param max_x: Max world x coordinate
        :param max_y: Max world y coordinate
        :return: occupancy map
        """

        scale = min(max_x, max_y)
        robot_range = max(1, int(math.floor(map_size * robot_size / scale / 2)))

        occupancy_map = (
            np.ones((map_size + 2 * robot_range, map_size + 2 * robot_range)) * 255
        )
        try:
            for world_points in position_list_local:

                map_points = self.world_to_map(world_points, map_size, max_x, max_y)
                if map_points == None:
                    continue
                x = map_points[0]
                y = map_points[1]
                for m in range(-robot_range, robot_range, 1):
                    for n in range(-robot_range, robot_range, 1):
                        occupancy_map[x + m][y + n] = 0
        except:
            pass
        occupancy_map = occupancy_map[robot_range:-robot_range, robot_range:-robot_range]

        return occupancy_map


    def generate_maps(self,
        position_lists_local,
    ):

        """
        Generate occupancy map
        :param position_lists_local: All robots' map coordinate
        :param self_orientation_list: All robots' orientation (map coordinate)
        :return: A list of occupancy maps
        """
        maps = []
        for robot_index in range(len(position_lists_local)):
            occupancy_map = self.generate_map_one(
                position_lists_local[robot_index],
                self.robot_size,
                self.map_size,
                self.max_x,
                self.max_y,
            )
            maps.append(occupancy_map)
        return np.array(maps)

# print(math.sin(arctan(-1.732,-1)))