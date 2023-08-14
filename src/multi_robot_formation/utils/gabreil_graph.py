"""
Some functions for other classes/functions
author: Xinchi Huang
"""

import numpy as np
import math
def is_valid_point(point_local,sensor_range=5,sensor_view_angle=2*math.pi/3):
    # print(point_local)
    point_local=np.array(point_local[:2])
    if np.linalg.norm(point_local)>sensor_range:
        return False
    if abs(math.atan2(point_local[1], point_local[0]))> sensor_view_angle / 2:
        return False
    # print("valid",point_local)
    return True

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
    for i in range(len(position_lists_global)):
        x_self = position_lists_global[i][0]
        y_self = position_lists_global[i][1]
        position_list_local_i = []
        for j in range(len(position_lists_global)):
            if i == j:
                position_list_local_i.append([0,0,0])
                continue
            point_local_raw = [
                position_lists_global[j][0] - x_self,
                position_lists_global[j][1] - y_self,
                0
            ]
            point_local_rotated = rotation(
                point_local_raw, self_orientation_global[i]
            )
            position_list_local_i.append(point_local_rotated)
        position_lists_local.append(position_list_local_i)

    return np.array(position_lists_local), np.array(self_orientation_global)



#### not finished
def get_gabreil_graph_local(position_array,view_range=5,view_angle=2*math.pi/3):
    position_array = np.array(position_array)
    self_orientation_global=position_array[:,2]
    position_array_local, self_orientation_global=global_to_local(position_array, self_orientation_global)
    # print(position_array_local)
    node_num=position_array_local.shape[0]
    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    self_position=np.zeros((1,2))
    for u in range(node_num):
        for v in range(node_num):
            if u==v:
                continue
            m = (position_array_local[u][v][:2] + self_position) / 2
            if is_valid_point(position_array_local[u][v][:2],view_range,view_angle)==False:
                gabriel_graph[u][v] = 0
                continue
            for w in range(node_num):
                if w == v:
                    continue
                if np.linalg.norm(position_array_local[u][w][:2] - m) < np.linalg.norm(
                    position_array_local[u][v][:2] - m
                ):
                    gabriel_graph[u][v] = 0
                    break
    return gabriel_graph

pose_list = [[-1.1025258903651642, 4.083962719940828, -2.6169047514337658],
             [0.46001556148439104, 2.783759094734095, 2.2540141643040292],
             [-1.3235583525190635, 1.8790783352091525, 1.4575483409100227],
             [2.394473099639181, 2.2757565243225466, 2.8741641810774223],
             [-2.8698225499017735, 3.147596063480031, -0.7098448661311878]]

#
# pose_list = [[-4, 0, math.pi / 2],
#              [4, 0, -math.pi / 2],
#              [-2, -2, 0],
#              [-2, 2, 0],
#              [2, 2, 0],
#              # [3, -3, 0],
#              # [0, 0, 0],
#              ]

graph=get_gabreil_graph_local(pose_list)
for l in graph:
    print(l)