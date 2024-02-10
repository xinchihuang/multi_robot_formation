"""
Codes for plot experiment results
author: Xinchi Huang
"""

import os
import sys

print(os.getcwd())
import os
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# from LocalExpertController import LocalExpertController
from utils.gabreil_graph import get_gabreil_graph
import matplotlib.ticker as ticker

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
                if w == v or w==u:
                    continue
                if np.linalg.norm(position_array[w] - m) <= np.linalg.norm(
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
    # print(distance_dict)
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


def plot_relative_distance_gabreil(dt, pose_array, save_path='',sensor_range=2):
    """
    Plot line chart for robots relative distance, Only show the distance which are
    edges of gabreil graph
    :param dt: Time interval
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    position_array = pose_array[:, -1, :]
    gabriel_graph = get_gabreil_graph(position_array,sensor_range=sensor_range)
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
    plt.figure(figsize=(10, 6))
    for key, _ in distance_dict.items():
        plt.plot(xlist, distance_dict[key], label=key,linewidth=5)
    # plt.legend()
    plt.subplots_adjust(left=0.05,
                        bottom=0.18,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)", fontsize=30)
    plt.ylabel("distance(m)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png"), pad_inches=0.0)
    plt.close()
def plot_relative_distance_gabreil_real(dt, pose_array, save_path='',sensor_range=2):
    """
    Plot line chart for robots relative distance, Only show the distance which are
    edges of gabreil graph
    :param dt: Time interval
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    position_array = pose_array[:, -1, :]
    gabriel_graph = get_gabreil_graph(position_array,sensor_range=sensor_range)
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
    plt.figure(figsize=(10, 6))
    for key, _ in distance_dict.items():
        plt.plot(xlist, distance_dict[key], label=key,linewidth=5)
    # plt.legend()
    plt.ylim(0,3)
    plt.subplots_adjust(left=0.13,
                        bottom=0.16,
                        right=0.98,
                        top=0.95,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)", fontsize=30)
    plt.ylabel("distance(m)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_real_" + str(rob_num) + ".png"), pad_inches=0.0)
    plt.close()


def plot_formation_gabreil(pose_array,save_path='',file_name="formation_gabreil.png",desired_distance=1,xlim=1.5,ylim=1.5,sensor_range=2,robot_size=0.1):
    """
        Plot the formation of robots, plot the gabreil graph
        :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
        :param save_path: Path to save figures
        :return:
        """
    rob_num = np.shape(pose_array)[0]
    position_array = pose_array[:, -1, :]
    gabriel_graph = get_gabreil_graph(position_array, sensor_range=sensor_range)
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot for position_array
    ax.scatter(position_array[:, 0], position_array[:, 1],s=100,c="black")

    formation_error = 0
    count = 0
    for i in range(rob_num):
        circle = plt.Circle((position_array[i][0], position_array[i][1]), robot_size, color='black', fill=False)
        ax.add_artist(circle)
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            if distance > 5:
                continue
            ax.plot(xlist, ylist, label=f"Distance: {distance: .2f}",linewidth=10)
            count += 1
            formation_error += abs(distance - desired_distance)


    # ax.set_title(f"Average formation error: {formation_error / count:.5f}", fontsize=20)
    ax.set_xlabel("x(m)", fontsize=35)
    ax.set_ylabel("y(m)", fontsize=35)
    # ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.grid()
    ax.set_aspect('equal',adjustable='box')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Save the figure
    fig.savefig(os.path.join(save_path, file_name), pad_inches=0.0)
    plt.close(fig)
def plot_formation_gabreil_real(pose_array,save_path='',desired_distance=2,xlim=4,ylim=4,sensor_range=2,robot_size=0.1):
    """
    Plot the formation of robots, plot the gabreil graph
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    position_array = pose_array[:, -1, :]
    gabriel_graph = get_gabreil_graph(position_array, sensor_range=sensor_range)
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 10))
    # Scatter plot for position_array
    ax.scatter(position_array[:, 0], position_array[:, 1])

    formation_error = 0
    count = 0
    for i in range(rob_num):
        circle = plt.Circle((position_array[i][0], position_array[i][1]), robot_size, color='black', fill=False)
        ax.add_artist(circle)
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            if distance > 5:
                continue
            ax.plot(xlist, ylist, label=f"Distance: {distance: .2f}")
            count += 1
            formation_error += abs(distance - desired_distance)


    ax.set_aspect('equal')
    ax.set_title(f"Average formation error: {formation_error / count:.5f}", fontsize=20)
    ax.set_xlabel("x(m)", fontsize=30)
    ax.set_ylabel("y(m)", fontsize=30)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.grid()

    # Save the figure
    fig.savefig(os.path.join(save_path, f"formation_gabreil_real_{rob_num}.png"), pad_inches=0.0)
    plt.close(fig)

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
def plot_triangle(ax,pos,theta,length,color):
    x=pos[0]
    y=pos[1]
    p1=[x+2*length*math.cos(theta),y+2*length*math.sin(theta)]
    p2=[x+length*math.cos(theta-2*math.pi/3),y+length*math.sin(theta-2*math.pi/3)]
    p3 = [x + length * math.cos(theta + 2*math.pi / 3), y + length * math.sin(theta + 2*math.pi / 3)]
    # ax.scatter(x,y,c=color)
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=color,linewidth=4)
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]],color=color,linewidth=4)
    ax.plot([p3[0],p1[0]],[p3[1],p1[1]],color=color,linewidth=4)
def plot_trace_triangle(pose_array,save_path='',time_step=1000,xlim=8,ylim=8,sensor_range=2):
    rob_num = np.shape(pose_array)[0]
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    fig,ax=plt.subplots(figsize=(10, 10))
    for i in range(rob_num):
        color = next(colors)
        xtrace = []
        ytrace = []
        for p in range(0,time_step-1,500):
            pos=[pose_array[i][p][0],pose_array[i][p][1]]
            theta=pose_array[i][p][2]
            plot_triangle(ax, pos,theta, 0.1, color)
            xtrace.append(pose_array[i][p][0])
            ytrace.append(pose_array[i][p][1])
            ax.plot(xtrace,ytrace,color=color,linestyle='--')
    # position_array = pose_array[:, 0, :]

    position_array = pose_array[:, time_step - 1, :2]
    gabriel_graph = get_gabreil_graph(position_array,sensor_range=sensor_range)

    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            ax.plot(xlist, ylist,color="black",linewidth=4)
    plt.subplots_adjust(left=0.18,
                        bottom=0.13,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("x(m)", fontsize=30)
    plt.ylabel("y(m)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_" + str(rob_num) + ".png"))
    plt.close()
    # plt.show()
def plot_trace_triangle_real(pose_array,save_path='',time_step=5000,xlim=8,ylim=8,sensor_range=2):
    rob_num = np.shape(pose_array)[0]
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    fig,ax=plt.subplots(figsize=(10, 10))
    for i in range(rob_num):
        color = next(colors)
        xtrace = []
        ytrace = []
        for p in range(0,time_step+1,500):
            pos=[pose_array[i][p][0],pose_array[i][p][1]]
            theta=pose_array[i][p][2]
            plot_triangle(ax, pos,theta, 0.1, color)
            xtrace.append(pose_array[i][p][0])
            ytrace.append(pose_array[i][p][1])
            ax.plot(xtrace,ytrace,color=color,linestyle='--')
    # position_array = pose_array[:, 0, :]

    position_array = pose_array[:, time_step - 1, :2]
    gabriel_graph = get_gabreil_graph(position_array,sensor_range=sensor_range)

    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            ax.plot(xlist, ylist,color="black",linewidth=4)
    plt.subplots_adjust(left=0.18,
                        bottom=0.13,
                        right=0.95,
                        top=0.95,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("x(m)", fontsize=30)
    plt.ylabel("y(m)", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_real_" + str(rob_num) + ".png"))
    plt.close()
    # plt.show()

def plot_load_data(root_dir,dt=0.05):
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
        trace_array_single = np.load(os.path.join(root_dir, robot_path, "trace.npy"),allow_pickle=True)
        trace_array_single = np.expand_dims(trace_array_single, axis=0)
        if isinstance(trace_array, type(None)):
            trace_array = trace_array_single
            continue
        trace_array = np.concatenate((trace_array, trace_array_single), axis=0)
    print(trace_array.shape)
    position_array = trace_array[:, :, :2]
    orientation_array=trace_array[:, :, 2]
    # pose_array=np.concatenate(position_array,orientation_array,axis=3)

    plot_relative_distance(dt, position_array, root_dir)
    plot_relative_distance_gabreil(dt, position_array, root_dir)
    # plot_formation_gabreil(position_array,orientation_array, root_dir)
    # plot_trace(position_array, root_dir)
    # print(orientation_array)
    # plot_trace_triangle(position_array ,orientation_array, root_dir)
    velocity_array = None
    for robot_path in robot_path_list:
        velocity_array_single = np.load(
            os.path.join(root_dir, robot_path, "control.npy"),allow_pickle=True
        )
        velocity_array_single = np.expand_dims(velocity_array_single, axis=0)
        if isinstance(velocity_array, type(None)):
            velocity_array = velocity_array_single
            continue
        velocity_array = np.concatenate((velocity_array, velocity_array_single), axis=0)
    plot_wheel_speed(dt, velocity_array, root_dir)
def plot_load_data_gazebo(root_dir,desired_distance=1,sensor_range=2,dt=0.05):
    """

    :param dt: Time interval
    :param dir: Root dir
    :return:
    """
    position_array = np.load(os.path.join(root_dir, "trace.npy"))
    orientation_array=position_array[:,:,2]

    position_array=np.transpose(position_array,(1,0,2))
    # plot_relative_distance(dt, position_array, root_dir)
    plot_relative_distance_gabreil(dt, position_array, root_dir, sensor_range=sensor_range)
    plot_formation_gabreil(position_array, root_dir,desired_distance=desired_distance,sensor_range=sensor_range)
    plot_trace_triangle(position_array,root_dir,sensor_range=sensor_range)

def plot_load_data_gazebo_multi_fromation(root_dir, desired_distance=1, sensor_range=2, num_graph=4):
        """

        :param dt: Time interval
        :param dir: Root dir
        :return:
        """
        pose_array = np.load(os.path.join(root_dir, "trace.npy"))
        pose_array = np.transpose(pose_array, (1, 0, 2))

        range=int((pose_array.shape[1]-1)/(num_graph-1))
        print(pose_array.shape,range)
        i=0
        while i<num_graph:
            plot_formation_gabreil(pose_array[:,:range*i+1,:], root_dir,file_name=f"formation_gabreil_{pose_array.shape[0]}_{i}.png", desired_distance=desired_distance, sensor_range=sensor_range)
            i += 1
        plot_trace_triangle_real(pose_array, time_step=pose_array.shape[1], xlim=1.5, ylim=1.5, save_path=str(root_dir))
        plot_formation_gabreil_real(pose_array, desired_distance=1.1, xlim=2, ylim=2, save_path=str(root_dir))
        plot_relative_distance_gabreil_real(0.01, pose_array, save_path=str(root_dir))
#
if __name__ == "__main__":

    # plot_load_data_gazebo("/home/xinchi/gazebo_data/ViT_5_full/70")
    root_path="C:\\Users\\huang xinchi\\Desktop\\multi_robot_formation\\scripts\\plots"
    for path in os.listdir(root_path):
        plot_load_data_gazebo_multi_fromation(os.path.join(root_path,path),desired_distance=1.1,sensor_range=2,num_graph=4)
    # trace_array=np.load("/home/xinchi/gazebo_data/0/trace.npy")
    # print(trace_array.shape)