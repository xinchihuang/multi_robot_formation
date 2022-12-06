import occupancy_map_simulator
import math
import random
import numpy as np
from collections import defaultdict
import cv2
import os

class ControlData:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        # self.omega_left = 0
        # self.omega_right = 0

        self.velocity_x = 0
        self.velocity_y = 0
class SensorData:
    """
    A class for record sensor data
    """

    def __init__(self):
        self.robot_index = None
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.occupancy_map = None
class SceneData:
    """
    A class for passing data from scene
    """

    def __init__(self):
        self.observation_list = None
        self.adjacency_list = None
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
def update_adjacency_list(position_list):
    """
    Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

    """
    node_num=len(position_list)
    position_array = np.array(position_list)

    # Get Gabreil Graph
    gabriel_graph = get_gabreil_graph(position_array, node_num)

    # Create adjacency list
    new_adj_list = defaultdict(list)
    for i in range(node_num):
        for j in range(node_num):
            if gabriel_graph[i][j] == 1 and not i == j:
                distance = (
                    (position_array[i][0] - position_array[j][0]) ** 2
                    + (position_array[i][1] - position_array[j][1]) ** 2
                ) ** 0.5
                new_adj_list[i].append(
                    (
                        j,
                        position_array[j][0],
                        position_array[j][1],
                        distance,
                    )
                )
    return  new_adj_list
def centralized_control(index, sensor_data, scene_data, desired_distance):

    out_put = ControlData()
    if not scene_data:
        out_put.velocity_x = 0
        out_put.velocity_y = 0
        return out_put

    self_position = sensor_data.position
    # self_orientation = sensor_data.orientation
    self_x = self_position[0]
    self_y = self_position[1]
    neighbors = scene_data.adjacency_list[index]
    # print(neighbors)
    velocity_sum_x = 0
    velocity_sum_y = 0
    for neighbor in neighbors:
        rate = (neighbor[3] - desired_distance) / neighbor[3]
        velocity_x = rate * (self_x - neighbor[1])
        velocity_y = rate * (self_y - neighbor[2])
        velocity_sum_x -= velocity_x
        velocity_sum_y -= velocity_y
    # transform speed to wheels speed
    theta = sensor_data.orientation[2]
    out_put.robot_index = index
    out_put.velocity_x = velocity_sum_x
    out_put.velocity_y = velocity_sum_y

    return out_put
def generate(number_of_robot,max_disp_range,min_disp_range,desired_distance):
    global_pose_list = []
    self_orientation_list=[]
    for i in range(number_of_robot):
        while True:
            alpha = math.pi * (2 * random.random())
            rho = max_disp_range * random.random()
            pos_x = rho * math.cos(alpha)
            pos_y = rho * math.sin(alpha)
            theta = 2 * math.pi * random.random()
            too_close = False
            for p in global_pose_list:
                if (pos_x - p[0]) ** 2 + (pos_y - p[1]) ** 2 <= min_disp_range**2:
                    too_close = True
                    break
            if too_close:
                continue
            global_pose_list.append([pos_x, pos_y, 0])
            self_orientation_list.append(theta)
            break
    position_lists_local, self_pose = occupancy_map_simulator.global_to_local(global_pose_list)
    occupancy_maps=occupancy_map_simulator.generate_maps(position_lists_local,self_orientation_list)
    ref_control_list=[]
    adjacency_lists=[]
    for i in range(number_of_robot):
        adjacency_list_i=update_adjacency_list(global_pose_list)
        adjacency_lists.append(adjacency_list_i)
        sensor_data_i=SensorData()
        sensor_data_i.position=global_pose_list[i]
        sensor_data_i.orientation=[0,0,global_pose_list[i][2]]
        scene_data_i=SceneData()
        scene_data_i.adjacency_list=adjacency_list_i
        control_i=centralized_control(i, sensor_data_i, scene_data_i,desired_distance)
        ref_control_list.append([control_i.velocity_x,control_i.velocity_y])
        #

    return occupancy_maps,ref_control_list,adjacency_lists

present = os.getcwd()
root = os.path.join(present, "data")
if not os.path.exists(root):
    os.mkdir(root)

i=0
while i<10000:
    if i%100==0:
        print(i)
    num_dirs = len(os.listdir(root))
    data_path = os.path.join(root, str(num_dirs))
    try:
        occupancy_maps, ref_control_list,adjacency_lists=generate(5,5,0.2,1)
        occupancy_maps_array=np.array(occupancy_maps)
        ref_control_array=np.array(ref_control_list)
        adjacency_lists_array=np.array(adjacency_lists)
        os.mkdir(data_path)
        np.save(os.path.join(data_path,"occupancy_maps"),occupancy_maps_array)
        np.save(os.path.join(data_path, "reference_controls"), ref_control_array)
        np.save(os.path.join(data_path, "adjacency_lists"), adjacency_lists_array)
        i+=1
    except:
        continue
