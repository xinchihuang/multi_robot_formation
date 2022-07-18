"""
Codes for plot experiment results
author: Xinchi Huang
"""
import os
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def gabriel(pose_array):
    """
    Get the gabriel graph of the formation
    :param pose_array: A numpy array contains all robots formation
    :return: Gabriel graph ( 2D matrix ) 1 represent connected, 0 represent disconnected
    """
    node_mum = np.shape(pose_array)[0]
    gabriel_graph = [[1] * node_mum for _ in range(node_mum)]
    position_array = pose_array[:, -1, :2]
    for u in range(node_mum):
        for v in range(node_mum):
            m = (position_array[u] + position_array[v]) / 2
            for w in range(node_mum):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(
                    position_array[u] - m
                ):
                    gabriel_graph[u][v] = 0
                    gabriel_graph[v][u] = 0
                    break
    return gabriel_graph


def plot_wheel_speed(dt, velocity_array, save_path):
    """
    Plot line chart for robots wheel speeds
    :param dt: Time interval
    :param velocity_array: Robots velocity data 3D numpy array [robot:[time step:[left,right]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(velocity_array)[0]
    xlist = []
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    for i in range(np.shape(velocity_array)[1]):
        xlist.append(i * dt)
    plt.figure(figsize=(10, 10))
    for i in range(rob_num):
        color = next(colors)
        plt.plot(
            xlist,
            velocity_array[i, :, 0],
            color=color,
            label="Robot " + str(i) + " left wheel speed",
        )
        plt.plot(
            xlist,
            velocity_array[i, :, 1],
            "--",
            color=color,
            label="Robot " + str(i) + " right wheel speed",
        )
    # plt.legend()
    plt.title("Wheel speeds")
    plt.xlabel("time(s)")
    plt.ylabel("velocity(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "wheel_speed_" + str(rob_num) + ".png"))
    plt.close()
    # plt.show()


def plot_relative_distance(dt, pose_array, save_path):
    """
    Plot line chart for robots relative distance
    :param dt: Time interval
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    distance_dict = {}
    xlist = []
    for i in range(np.shape(pose_array)[1]):
        xlist.append(i * dt)
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            name = str(i + 1) + " to " + str(j + 1)
            distance_array = np.sqrt(
                np.square(pose_array[i, :, 0] - pose_array[j, :, 0])
                + np.square(pose_array[i, :, 1] - pose_array[j, :, 1])
            )
            distance_dict[name] = distance_array
    plt.figure(figsize=(10, 10))
    for key, _ in distance_dict.items():
        plt.plot(xlist, distance_dict[key], label=key)
    # plt.legend()
    plt.title("Relative distance")
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_" + str(rob_num) + ".png"))
    plt.close()


def plot_relative_distance_gabreil(dt, pose_array, save_path):
    """
    Plot line chart for robots relative distance, Only show the distance which are
    edges of gabreil graph
    :param dt: Time interval
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    distance_dict = {}
    xlist = []
    for i in range(np.shape(pose_array)[1]):
        xlist.append(i * dt)
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            name = str(i + 1) + " to " + str(j + 1)
            distance_array = np.sqrt(
                np.square(pose_array[i, :, 0] - pose_array[j, :, 0])
                + np.square(pose_array[i, :, 1] - pose_array[j, :, 1])
            )
            distance_dict[name] = distance_array
    plt.figure(figsize=(10, 10))
    for key, _ in distance_dict:
        plt.plot(xlist, distance_dict[key], label=key)
    # plt.legend()
    plt.title("Relative distance gabreil")
    plt.xlabel("time(s)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(
        os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png")
    )
    plt.close()


def plot_formation_gabreil(pose_array, save_path):
    """
    Plot the formation of robots, plot the gabreil graph
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, -1, :2]
    plt.figure(figsize=(10, 10))
    plt.scatter(position_array[:, 0], position_array[:, 1])
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt(
                (xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2
            )
            plt.plot(xlist, ylist, label=f"Distane: {distance:8 .2f}")
    plt.legend()
    plt.title("Formation")
    plt.xlabel("distance(m)")
    plt.ylabel("distance(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "formation_gabreil_" + str(rob_num) + ".png"))
    plt.close()


def plot_trace(position_array, save_path):
    """
    Plot the trace(dots) of robots
    :param position_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(position_array)[0]

    colors = itertools.cycle(mcolors.TABLEAU_COLORS)

    plt.figure(figsize=(10, 10))
    for i in range(rob_num):
        color = next(colors)
        for p in range(np.shape(position_array)[1]):
            plt.scatter(position_array[i][p][0], position_array[i][p][1], s=10, c=color)
        plt.scatter(
            position_array[i][0][0], position_array[i][0][1], s=150, c=color, marker="x"
        )
    # plt.legend()
    plt.title("Trace")
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_" + str(rob_num) + ".png"))
    plt.close()


def plot_load_data(dt, root_dir):
    """

    :param dt: Time interval
    :param dir: Root dir
    :return:
    """
    robot_path_list = []
    for _, dirs, _ in os.walk(root_dir, topdown=False):
        for name in dirs:
            robot_path_list.append(name)
    trace_array = None
    for robot_path in robot_path_list:
        trace_array_single = np.load(os.path.join(root_dir, robot_path, "trace.npy"))
        trace_array_single = np.expand_dims(trace_array_single, axis=0)
        if isinstance(trace_array, type(None)):
            trace_array = trace_array_single
            continue
        trace_array = np.concatenate((trace_array, trace_array_single), axis=0)
    position_array = trace_array[:, :, 0, :2]
    plot_relative_distance(dt, position_array, root_dir)
    # plot_relative_distance_gabreil(dt, position_array, root_dir)
    # plot_formation_gabreil(position_array, root_dir)
    plot_trace(position_array, root_dir)
    velocity_array = None
    for robot_path in robot_path_list:
        velocity_array_single = np.load(
            os.path.join(root_dir, robot_path, "control.npy")
        )
        velocity_array_single = np.expand_dims(velocity_array_single, axis=0)
        if isinstance(velocity_array, type(None)):
            velocity_array = velocity_array_single
            continue
        velocity_array = np.concatenate((velocity_array, velocity_array_single), axis=0)
    plot_wheel_speed(dt, velocity_array, root_dir)


if __name__ == "__main__":
    plot_load_data(0.05, "/home/xinchi/multi_robot_formation/saved_data/2")
