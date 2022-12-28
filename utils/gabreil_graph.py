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
