import math
import numpy as np
from collections import defaultdict
import cv2
import os
import sys

from .gabreil_graph import get_gabreil_graph, get_gabreil_graph_local,global_to_local
from .occupancy_map_simulator import MapSimulator
from comm_data import ControlData, SensorData, SceneData
# from controller import Controller
import copy
import math
def sort_pose(position_list):
    global_pose_array = np.array(position_list)
    temp = copy.deepcopy(global_pose_array)
    orders = np.argsort(global_pose_array, axis=0)
    for i in range(len(orders)):
        global_pose_array[i, :] = temp[orders[i][0]]
    return global_pose_array


class DataGenerator:
    def __init__(self, desired_distance=2,max_x=5,max_y=5,local=True, partial=True,safe_margin=0.1,sensor_angle=math.pi/2,K_f=1,K_m=0,K_omega=0,max_speed=1,max_omega=0):
        self.desired_distance=desired_distance
        self.local = local
        self.partial = partial
        self.max_x = max_x
        self.max_y = max_y
        self.safe_margin=safe_margin
        self.K_f = K_f
        self.K_m = K_m
        self.K_omega = K_omega
        self.max_speed = max_speed
        self.max_omega = max_omega

        self.sensor_angle = sensor_angle
        self.map_simulator=MapSimulator(max_x=self.max_x,max_y=self.max_y,local=self.local, partial=self.partial,sensor_view_angle=self.sensor_angle)


    def update_adjacency_list(self, position_list):
        """
        Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

        """
        position_array = np.array(position_list)
        node_num = position_array.shape[0]
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
        return new_adj_list
    def update_neighbor(self, position_list):
        """
        Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

        """
        position_array = np.array(position_list)
        node_num = position_array.shape[0]
        gabriel_graph = get_gabreil_graph_local(position_array, node_num)
        return gabriel_graph[0]
    def generate_map_all(self, global_pose_array):
        """
        Generate a set of data for training
        :param global_pose_array: Robot's global pose, shape:(number of robot,3) [x,y,theta]
        :return: A list of local observations
        """

        # global_pose_array = [[-2, -2, math.pi/4],
        #          [-2, 2, -math.pi/4],
        #          [2, 2, -3*math.pi/4],
        #          [2, -2, 3*math.pi/4],
        #          [0, 0, 0]]
        global_pose_array = np.array(global_pose_array)
        # self_orientation_array = np.array(self_orientation_array)
        occupancy_map_simulator = self.map_simulator
        position_lists_local= global_to_local(global_pose_array)


        for i in range(len(position_lists_local)):
            while len(position_lists_local[i]) < len(position_lists_local) - 1:
                position_lists_local[i].append([float("inf"), float("inf"), 0])
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)
        ref_control_list = []
        adjacency_lists = []
        number_of_robot = global_pose_array.shape[0]
        controller = LocalExpertController(desired_distance=self.desired_distance,safe_margin=self.safe_margin,sensor_range=self.max_x,sensor_angle=self.sensor_angle,K_f=self.K_f,K_omega=self.K_omega,max_speed=self.max_speed,max_omega=self.max_omega)
        for robot_index in range(number_of_robot):
            control_i = controller.get_control(robot_index,global_pose_array)
            velocity_x, velocity_y,omega = control_i.velocity_x, control_i.velocity_y,control_i.omega
            ref_control_list.append([velocity_x, velocity_y,omega])
        # print(self.sensor_angle)
        # print(global_pose_array)
        # print(ref_control_list[4])
        # print(global_to_local(global_pose_array)[4])
        # cv2.imshow("robot view ", np.array(occupancy_maps[4]))
        # cv2.waitKey(0)
        # for i in range(len(position_lists_local)):
        #     position_lists_local[i] = sort_pose(np.array(position_lists_local[i]))
        # position_lists_local = np.array(position_lists_local)
        # #neighbor
        # neighbor_lists = []
        # number_of_robot = position_lists_local.shape[0]
        # for robot_index in range(number_of_robot):
        #     neighbor_list_i = self.update_neighbor(position_lists_local[robot_index])
        #     neighbor_lists.append(neighbor_list_i)
        #
        # # position
        # selected = position_lists_local[:, :, :2]
        # position_lists_squeezed = np.reshape(selected, (selected.shape[0], selected.shape[1] * selected.shape[2]))
        # position_lists_squeezed[position_lists_squeezed == float("inf")] = 0

        return (
            np.array(occupancy_maps),
            np.array(ref_control_list),
            np.array(adjacency_lists),
            np.array((1,1)),
            np.array((1, 1)),
            # np.array(position_lists_squeezed),
            # np.array(neighbor_lists),
        )
    def generate_pose_one(self, global_pose_array, self_orientation_array):
        global_pose_array = np.array(global_pose_array)
        number_of_robot = global_pose_array.shape[0]
        self_orientation_array = np.array(self_orientation_array)
        occupancy_map_simulator = self.map_simulator

        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            global_pose_array, self_orientation_array
        )
        position_array_local = np.zeros((number_of_robot, number_of_robot - 1, 3))
        position_array_local[:, :, 2] = -1
        for i in range(len(position_lists_local)):
            for j in range(len(position_lists_local[i])):
                position_array_local[i][j] = position_lists_local[i][j]
        ref_control_list = []
        adjacency_lists = []

        for robot_index in range(number_of_robot):

            adjacency_list_i = self.update_adjacency_list(global_pose_array)
            adjacency_lists.append(adjacency_list_i)
            sensor_data_i = SensorData()
            sensor_data_i.position = global_pose_array[robot_index]
            sensor_data_i.orientation = [0, 0, self_orientation_array[robot_index]]
            scene_data_i = SceneData()
            scene_data_i.adjacency_list = adjacency_list_i

            controller = CentralizedController()
            # print("robot_index",robot_index)
            # print(position_lists_local[robot_index])
            # print(adjacency_list_i)
            control_i=controller.get_control(robot_index,adjacency_list_i[robot_index],global_pose_array[robot_index])

            velocity_x, velocity_y = control_i.velocity_x, control_i.velocity_y
            if self.local:

                theta = self_orientation_array[robot_index]
                velocity_x_global = velocity_x * math.cos(
                    theta
                ) + velocity_y * math.sin(theta)
                velocity_y_global = -velocity_x * math.sin(
                    theta
                ) + velocity_y * math.cos(theta)
                velocity_x = velocity_x_global
                velocity_y = velocity_y_global

            ref_control_list.append([velocity_x, velocity_y])
        return (

            np.array(position_array_local),
            np.array(self_orientation),
            np.array(ref_control_list),
            np.array(adjacency_lists),

        )


if __name__ == "__main__":
    # self_pose_array=[[-3, -3, 0], [-3, 3, 0], [3, 3, 0], [3, -3, 0], [0, 0, 0]]
    # self_orientation_array=[0,0,0,0,0]
    # data_generator=DataGenerator(partial=False)
    #
    # occupancy_maps,ref_control_lists,adjacency_lists,position_lists_squeezed,neighbor_lists=data_generator.generate_map_all(self_pose_array,self_orientation_array)
    # print(ref_control_lists)
    # cv2.imshow("robot view (all)", occupancy_maps[0])
    # cv2.waitKey(0)
    # occupancy_maps, reference, adjacency_lists = data_generator.generate_map_control(
    #     self_pose_array, self_orientation_array
    # )
    # print(reference)
    # cv2.imshow("robot view (control)", occupancy_maps[0])
    # cv2.waitKey(0)
    pass

