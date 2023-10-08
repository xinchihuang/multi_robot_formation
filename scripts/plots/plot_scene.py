"""
Codes for plot experiment results
author: Xinchi Huang
"""

import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/model")
print(sys.path)


import os
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# from LocalExpertController import LocalExpertController

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

# def plot_speed(dt, pose_array, save_path):
#     """
#     Plot line chart for robots wheel speeds
#     :param dt: Time interval
#     :param velocity_array: Robots velocity data 3D numpy array [robot:[time step:[left,right]]]
#     :param save_path: Path to save figures
#     :return:
#     """
#     print(pose_array.shape)
#     controller = LocalExpertController()
#     rob_num = np.shape(pose_array)[0]
#     gabriel_graph = gabriel(pose_array)
#     distance_dict = {}
#     speed_dict={}
#     xlist = []
#     for i in range(np.shape(pose_array)[1]):
#         xlist.append(i * dt)
#     for i in range(np.shape(pose_array)[1]):
#         position_array=pose_array[:,i,:]
#         for j in range(rob_num):
#             control=controller.get_control(j,position_array)
#             name_x="x_"+str(j)
#             name_y="y_"+str(j)
#             if not name_x in speed_dict:
#                 speed_dict[name_x]=[]
#             if not name_y in speed_dict:
#                 speed_dict[name_y]=[]
#             speed_dict[name_x].append(control.velocity_x)
#             speed_dict[name_y].append(control.velocity_y)
#     plt.figure(figsize=(5, 3))
#     for key, _ in speed_dict.items():
#         plt.plot(xlist, speed_dict[key], label=key)
#     # plt.legend()
#     plt.subplots_adjust(left=0.13,
#                         bottom=0.23,
#                         right=0.98,
#                         top=0.98,
#                         wspace=0.0,
#                         hspace=0.0)
#     plt.xlabel("time(s)", fontsize=20)
#     plt.ylabel("distance(m)", fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.grid()
#     plt.savefig(os.path.join(save_path, "speed_" + str(rob_num) + ".png"), pad_inches=0.0)
#     plt.close()
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
    print(distance_dict)
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
    plt.figure(figsize=(5, 3))
    for key, _ in distance_dict.items():
        plt.plot(xlist, distance_dict[key], label=key)
    # plt.legend()
    plt.subplots_adjust(left=0.13,
                        bottom=0.23,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("time(s)", fontsize=20)
    plt.ylabel("distance(m)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig(os.path.join(save_path, "relative_distance_gabreil_" + str(rob_num) + ".png"), pad_inches=0.0)
    plt.close()


def plot_formation_gabreil(pose_array,orientation_array, save_path,desired_distance=2):
    """
    Plot the formation of robots, plot the gabreil graph
    :param pose_array: Robots trace data 3D numpy array [robot:[time step:[x,y]]]
    :param save_path: Path to save figures
    :return:
    """
    rob_num = np.shape(pose_array)[0]
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, -1, :]
    print(position_array)
    plt.figure(figsize=(10, 10))
    plt.scatter(position_array[:, 0], position_array[:, 1])
    # for i in range(len(position_array)):
    #     plt.plot([position_array[i][0], position_array[i][0] + math.cos(position_array[i][2] + math.pi / 4)],
    #              [position_array[i][1], position_array[i][1] + math.sin(position_array[i][2] + math.pi / 4)],color="gray")
    #     plt.plot([position_array[i][0], position_array[i][0] + math.cos(position_array[i][2] - math.pi / 4)],
    #              [position_array[i][1], position_array[i][1] + math.sin(position_array[i][2] - math.pi / 4)],color="gray")

    xlist=[]
    ylist=[]
    formation_error=0
    count=0
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt(
                (xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2
            )
            if distance>5:
                continue
            plt.plot(xlist, ylist, label=f"Distane: {distance: .2f}")
            count+=1
            formation_error+=abs(distance-desired_distance)

    plt.subplots_adjust(left=0.16,
                        bottom=0.16,
                        right=0.95,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.title("Average formation error: "+str(formation_error/count))
    plt.xlabel("x(m)", fontsize=20)
    plt.ylabel("y(m)", fontsize=20)
    plt.legend()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.savefig(os.path.join(save_path, "formation_gabreil_" + str(rob_num) + ".png"), pad_inches=0.0)
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
def plot_triangle(ax,pos,theta,length,color):
    x=pos[0]
    y=pos[1]
    p1=[x+2*length*math.cos(theta),y+2*length*math.sin(theta)]
    p2=[x+length*math.cos(theta-2*math.pi/3),y+length*math.sin(theta-2*math.pi/3)]
    p3 = [x + length * math.cos(theta + 2*math.pi / 3), y + length * math.sin(theta + 2*math.pi / 3)]
    # ax.scatter(x,y,c=color)
    ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color=color)
    ax.plot([p2[0],p3[0]],[p2[1],p3[1]],color=color)
    ax.plot([p3[0],p1[0]],[p3[1],p1[1]],color=color)
def plot_trace_triangle(pose_array,orientation_array,save_path,stop_time=50):
    rob_num = np.shape(pose_array)[0]
    colors = itertools.cycle(mcolors.TABLEAU_COLORS)
    fig,ax=plt.subplots(figsize=(5, 5))
    for i in range(rob_num):
        color = next(colors)
        xtrace = []
        ytrace = []

        for p in range(0,stop_time*20-1,100):
            pos=pose_array[i][p]
            theta=orientation_array[i][p]
            plot_triangle(ax, pos,theta, 0.3, color)
            xtrace.append(pose_array[i][p][0])
            ytrace.append(pose_array[i][p][1])
            ax.plot(xtrace,ytrace,color=color,linestyle='--')
    gabriel_graph = gabriel(pose_array)
    position_array = pose_array[:, stop_time*20-1, :2]
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            ax.plot(xlist, ylist,color="black")

    plt.subplots_adjust(left=0.13,
                        bottom=0.11,
                        right=0.98,
                        top=0.98,
                        wspace=0.0,
                        hspace=0.0)
    plt.xlabel("x(m)", fontsize=15)
    plt.ylabel("y(m)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid()
    plt.savefig(os.path.join(save_path, "robot_trace_" + str(rob_num) + ".png"))
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
    print(position_array)

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
def plot_load_data_gazebo(root_dir,dt=0.05):
    """

    :param dt: Time interval
    :param dir: Root dir
    :return:
    """
    position_array = np.load(os.path.join(root_dir, "trace.npy"))
    orientation_array=position_array[:,:,2]

    position_array=np.transpose(position_array,(1,0,2))
    plot_relative_distance(dt, position_array, root_dir)
    plot_relative_distance_gabreil(dt, position_array, root_dir)
    plot_formation_gabreil(position_array,orientation_array, root_dir)
    # plot_speed(dt, position_array, root_dir)


if __name__ == "__main__":

    # plot_load_data_gazebo("/home/xinchi/gazebo_data/")
    root_path="/home/xinchi/gazebo_data/ViT_9_full"
    for path in os.listdir(root_path):
        plot_load_data_gazebo(os.path.join(root_path,path))
    # trace_array=np.load("/home/xinchi/gazebo_data/0/trace.npy")
    # print(trace_array.shape)