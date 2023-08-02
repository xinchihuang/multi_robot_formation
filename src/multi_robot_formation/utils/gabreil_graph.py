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
    :param node_num: number of robots
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


    return position_lists_local, self_orientation_global

#### not finished
def get_gabreil_graph_local(position_array):

    position_array = np.array(position_array)
    self_orientation_global=position_array[:,2]
    position_lists_local, self_orientation_global=global_to_local(position_array, self_orientation_global)
    print(position_lists_local)
    node_num=position_array.shape[0]

    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    for i in range(1,node_num):
        for j in range(1,node_num):
            gabriel_graph[i][j]=0
    self_position=np.zeros((1,3))
    for u in range(node_num):
        if position_array[u][0]==float("inf") or position_array[u][1]==float("inf"):
            continue
        m = (position_array[u] + self_position) / 2
        for w in range(node_num):
            if w == u:
                continue
            if np.linalg.norm(position_array[w] - m) < np.linalg.norm(
                position_array[u] - m
            ):
                gabriel_graph[u][0] = 0
                gabriel_graph[0][u] = 0
                break
    return gabriel_graph
