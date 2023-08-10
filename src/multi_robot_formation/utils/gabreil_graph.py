"""
Some functions for other classes/functions
author: Xinchi Huang
"""

import numpy as np
import math

def get_gabreil_graph(position_array):
    """
    Return a gabreil graph of the scene
    :param position_array: A numpy array contains all robots' positions
    :return: A gabreil graph( 2D list)
    """
    position_array=np.array(position_array)[:,:2]
    node_num=len(position_array)
    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    for u in range(node_num):
        for v in range(node_num):
            m = (position_array[u] + position_array[v]) / 2
            for w in range(node_num):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(
                    position_array[u] - m
                ):
                    gabriel_graph[u][v] = 0
                    gabriel_graph[v][u] = 0
                    break
    return gabriel_graph

def rotation(world_point, self_orientation):
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
def global_to_local(position_lists_global, self_orientation_global):
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
                position_list_local_i.append([0,0,0])
                continue
            point_local_raw = [
                position_lists_global[j][0] - x_self,
                position_lists_global[j][1] - y_self,
                position_lists_global[j][2] - z_self,
            ]
            point_local_rotated = rotation(
                point_local_raw, self_orientation_global[i]
            )
            position_list_local_i.append(point_local_rotated)
        position_lists_local.append(position_list_local_i)


    return np.array(position_lists_local), np.array(self_orientation_global)



#### not finished
def get_gabreil_graph_local(position_array,view_range=5,view_angle=120):
    position_array = np.array(position_array)
    self_orientation_global=position_array[:,2]
    position_array_local, self_orientation_global=global_to_local(position_array, self_orientation_global)
    node_num=position_array_local.shape[0]
    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    self_position=np.zeros((1,3))
    for u in range(node_num):
        for v in range(node_num):
            if u==v:
                gabriel_graph[u][v] = 0
                continue
            m = (position_array_local[u][v] + self_position) / 2
            if np.linalg.norm(position_array_local[u][v]) > view_range:
                gabriel_graph[u][v] = 0
                continue
            if abs(np.degrees(math.atan2(position_array_local[u][v][1], position_array_local[u][v][0]))) > view_angle / 2:
                # print(position_array_local[u])
                # print(u,v)
                # print(np.degrees(math.atan2(position_array_local[u][v][1], position_array_local[u][v][0])))
                gabriel_graph[u][v] = 0
                continue
            for w in range(node_num):
                if w == v:
                    continue
                if np.linalg.norm(position_array_local[u][w] - m) < np.linalg.norm(
                    position_array_local[u][v] - m
                ):
                    gabriel_graph[u][v] = 0
                    break
    return gabriel_graph
