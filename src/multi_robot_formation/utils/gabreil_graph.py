"""
Some functions for other classes/functions
author: Xinchi Huang
"""

import numpy as np


def get_gabreil_graph(position_array, node_num):
    """
    Return a gabreil graph of the scene
    :param position_array: A numpy array contains all robots' positions
    :param node_num: number of robots
    :return: A gabreil graph( 2D list)
    """
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


def get_gabreil_graph_local(position_array, node_num):
    """
    Return a gabreil graph of the scene
    :param position_array: A numpy array contains all other robots' positions reative  to the given robot
    :param node_num: number of robots
    :return: A gabreil graph( 2D list)
    """
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
