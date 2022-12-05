"""
Code for generating demo occupancy map from given points
author: Xinchi Huang
"""

import math
import numpy as np
import cv2
def arctan(x,y):
    if x == 0 and y > 0:
        theta = math.pi / 2
    elif x == 0 and y < 0:
        theta = -math.pi / 2
    elif x == 0 and y == 0:
        theta=0
    else:
        theta = math.atan(y / x)
    return theta
def blocking(position_lists_local,robot_size=0.2):
    out_position_lists_local=[]
    for self_i in range(len(position_lists_local)):
        position_lists_i=[]
        for robot_j in range(len(position_lists_local[self_i])):
            x = position_lists_local[self_i][robot_j][0]
            y = position_lists_local[self_i][robot_j][1]
            theta=arctan(x,y)
            block=False
            for robot_k in range(len(position_lists_local[self_i])):
                if robot_k==robot_j:
                    continue
                x_k = position_lists_local[self_i][robot_k][0]
                y_k = position_lists_local[self_i][robot_k][1]
                if x**2+y**2<x_k**2+y_k**2:
                    continue
                x_k1 = x_k - (robot_size / 2) * math.sin(theta)
                y_k1 = y_k + (robot_size / 2) * math.cos(theta)

                x_k2 = x_k + (robot_size / 2) * math.sin(theta)
                y_k2 = y_k - (robot_size / 2) * math.cos(theta)

                theta_k_1 = arctan(y_k1 , x_k1)
                theta_k_2 = arctan(y_k2 , x_k2)

                if min(theta_k_1,theta_k_2)<theta<max(theta_k_1,theta_k_2):
                    block=True
            if block==False:
                position_lists_i.append(position_lists_local[self_i][robot_j])
        out_position_lists_local.append(position_lists_i)
    return out_position_lists_local


def global_to_local(position_lists_global):
    """
    Get each robot's observation from global absolute position
    :param position_lists_global: Global absolute position of all robots in the world
    :return: A list of local observations
    """
    position_lists_local = []
    self_pose_list=[]
    for i in range(len(position_lists_global)):
        x_self = position_lists_global[i][0]
        y_self = position_lists_global[i][1]
        z_self = position_lists_global[i][2]
        self_pose_list.append([x_self,y_self,z_self])
        position_list_local_i = []
        for j in range(len(position_lists_global)):
            if i == j:
                continue
            position_list_local_i.append(
                [
                    position_lists_global[j][0] - x_self,
                    position_lists_global[j][1] - y_self,
                    position_lists_global[j][2] - z_self,
                ]
            )
        position_lists_local.append(position_list_local_i)
    position_lists_local=blocking(position_lists_local,robot_size=0.2)
    return position_lists_local,self_pose_list


def data_filter(world_point, max_x, max_y, max_height, min_range):
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
    if x > max_x or x < -max_x or y > max_y or y < -max_y or z < -max_height:  #
        return None
    if x < min_range and y < min_range and x > -min_range and y > -min_range:
        return None
    return [x, y, z]

def rotation(world_point,self_orientation):
    x = world_point[0]
    y = world_point[1]
    z = world_point[2]
    theta=-self_orientation
    x_relative=math.cos(theta)*x+math.sin(theta)*y
    y_relative=-math.sin(theta)*x+math.cos(theta)*y
    return [x_relative,y_relative,z]


def world_to_map(world_point, map_size, max_x, max_y):
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
def flatten_maps(maps,map_size=100):
    out=[]
    for map in maps:
        out.append(map.reshape((1, map_size * map_size)))
    return out

def generate_maps(position_lists_local,self_orientation_list, robot_size=0.2, max_height=0.3, map_size=100, max_x=10, max_y=10):

    """
    Generate occupancy map
    :param position_lists_local: All robots' map coordinate
    :param self_orientation_list: All robots' orientation (map coordinate)
    :param robot_size: Size of robot in occupancy map
    :param max_height: points' horizontal range
    :param map_size: The size of occupancy map
    :param max_x: Max world x coordinate
    :param max_y: Max world y coordinate
    :return: A list of occupancy maps
    """

    maps = []
    scale = min(max_x, max_y)
    robot_range = max(1, int(math.floor(map_size * robot_size / scale / 2)))
    for robot_index in range(len(position_lists_local)):

        occupancy_map = (
            np.ones((map_size + 2 * robot_range, map_size + 2 * robot_range))*255
        )

        for world_points in position_lists_local[robot_index]:
            print(world_points)
            world_points_filtered = data_filter(
                world_points, max_x, max_y, max_height, 2 * robot_size
            )

            world_points_rotated=rotation(world_points_filtered,self_orientation_list[robot_index])
            map_points = world_to_map(world_points_rotated, map_size, max_x, max_y)
            if map_points == None:
                continue
            x = map_points[0]
            y = map_points[1]
            for m in range(-robot_range, robot_range, 1):
                for n in range(-robot_range, robot_range, 1):
                    occupancy_map[x + m][y + n] = 0
        occupancy_map = occupancy_map[
            robot_range:-robot_range, robot_range:-robot_range
        ]
        maps.append(occupancy_map)

    return maps


# global_positions=[[-4,-4,0],
#                 [-4,4,0],
#                 [4,4,0],
#                 [4,-4,0],
#                 [0,0,0]]
# position_lists_local,self_pose=global_to_local(global_positions)
# self_orientation_list=[math.pi/4,math.pi/4,0,0,0]
# robot_size,map_size,max_x,max_y=0.2 ,100,10,10
# max_height=0.3
# maps=generate_maps(position_lists_local,self_orientation_list,robot_size,max_height,map_size,max_x,max_y)
# map=maps[2]
# cv2.imshow("image",map)
# cv2.waitKey(0)
