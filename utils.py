"""
Some functions for other classes/functions
author: Xinchi Huang
"""

import numpy as np
from collections import defaultdict
def get_gabreil_graph(position_array,node_num):
    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    for u in range(node_num):
        for v in range(node_num):
            m = (position_array[u] + position_array[v]) / 2
            for w in range(node_num):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(position_array[u] - m):
                    gabriel_graph[u][v] = 0
                    gabriel_graph[v][u] = 0
                    break
    return gabriel_graph
def gabreil_graph(position_list):
    """

    :param position_list: [(robot_index,x,y),(robot_index,x,y)] A list contains all robots' position and index
    :return:
    """

    robot_num=len(position_list)
    graph=defaultdict(list)
    for u in range(robot_num):
        for v in range(robot_num):
            mid=(position_list[u][1]-position_list[v][1],position_list[u][2]-position_list[v][2])
            for w in range(robot_num):
                if w == v:
                    continue

