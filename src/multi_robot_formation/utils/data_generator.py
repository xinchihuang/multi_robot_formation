import math
import random
import numpy as np
from collections import defaultdict
import cv2
import os
import sys
print(os.getcwd())
from multi_robot_formation.utils.gabreil_graph import get_gabreil_graph, get_gabreil_graph_local
from multi_robot_formation.utils.occupancy_map_simulator import MapSimulator
from multi_robot_formation.comm_data import ControlData, SensorData, SceneData
# from controller import Controller
from multi_robot_formation.controller_new import *


class DataGenerator:
    def __init__(self, local=True, partial=True):
        self.local = local
        self.partial = partial

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

    def update_adjacency_list_lcoal(self, position_list_local, robot_index):
        """
        Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

        """
        position_array_local = np.array(position_list_local)
        position_array = np.concatenate(
            (np.array([[0, 0, 0]]), position_array_local), axis=0
        )
        node_num = position_array.shape[0]
        # Get Gabreil Graph
        gabriel_graph = get_gabreil_graph(position_array, node_num)
        # Create adjacency list
        new_adj_list = defaultdict(list)
        for j in range(node_num):
            if gabriel_graph[0][j] == 1 and not 0 == j:
                distance = (
                    (position_array[0][0] - position_array[j][0]) ** 2
                    + (position_array[0][1] - position_array[j][1]) ** 2
                ) ** 0.5
                new_adj_list[robot_index].append(
                    (
                        j,
                        position_array[j][0],
                        position_array[j][1],
                        distance,
                    )
                )
        return new_adj_list

    def generate_map_control(self, global_pose_array, self_orientation_array):
        global_pose_array = np.array(global_pose_array)
        self_orientation_array = np.array(self_orientation_array)
        occupancy_map_simulator = MapSimulator(local=self.local, partial=self.partial)

        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            global_pose_array, self_orientation_array
        )
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)
        ref_control_list = []
        adjacency_lists = []
        number_of_robot = global_pose_array.shape[0]
        for robot_index in range(number_of_robot):
            # if self.local:
            #     print("local")
            #     adjacency_list_i = self.update_adjacency_list_lcoal(position_lists_local[robot_index],robot_index)
            # else:
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
            # control_i = controller.centralized_control(
            #     robot_index,
            #     sensor_data_i,
            #     scene_data_i,
            # )

            control_i=controller.get_control(robot_index,adjacency_list_i[robot_index],global_pose_array[robot_index])

            velocity_x, velocity_y = control_i.velocity_x, control_i.velocity_y
            # print(velocity_x,velocity_y)
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
            np.array(occupancy_maps),
            np.array(ref_control_list),
            np.array(adjacency_lists),
        )
    def generate_map_graph(self, global_pose_array, self_orientation_array):
        global_pose_array = np.array(global_pose_array)
        self_orientation_array = np.array(self_orientation_array)
        occupancy_map_simulator = MapSimulator(local=self.local, partial=self.partial)

        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            global_pose_array, self_orientation_array
        )
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)
        neighbor_lists = []
        number_of_robot = global_pose_array.shape[0]
        for robot_index in range(number_of_robot):
            adjacency_list_i = self.update_adjacency_list(global_pose_array)
            neighbor_list_i=[0]*number_of_robot
            for n in adjacency_list_i[robot_index]:
                neighbor_list_i[n[0]]=1
            neighbor_lists.append(neighbor_list_i)
        return (
            np.array(occupancy_maps),
            np.array(neighbor_lists),
        )
    def generate_pose_one(self, global_pose_array, self_orientation_array):
        global_pose_array = np.array(global_pose_array)
        number_of_robot = global_pose_array.shape[0]
        self_orientation_array = np.array(self_orientation_array)
        occupancy_map_simulator = MapSimulator(local=self.local, partial=self.partial)

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

    # def generate(self,number_of_robot, max_disp_range, min_disp_range, desired_distance):
    #     global_pose_list = []
    #     self_orientation_list = []
    #     for i in range(number_of_robot):
    #         while True:
    #             alpha = math.pi * (2 * random.random())
    #             rho = max_disp_range * random.random()
    #             pos_x = rho * math.cos(alpha)
    #             pos_y = rho * math.sin(alpha)
    #             theta = 2 * math.pi * random.random()
    #             too_close = False
    #             for p in global_pose_list:
    #                 if (pos_x - p[0]) ** 2 + (pos_y - p[1]) ** 2 <= min_disp_range**2:
    #                     too_close = True
    #                     break
    #             if too_close:
    #                 continue
    #             global_pose_list.append([pos_x, pos_y, 0])
    #             self_orientation_list.append(theta)
    #             break
    #     occupancy_maps, ref_control_list, adjacency_lists = self.generate_one(global_pose_list, self_orientation_list)
    #     return occupancy_maps, ref_control_list, adjacency_lists


if __name__ == "__main__":
    self_pose_array=[[0,0,0],[-2,-2,0]]
    self_orientation_array=[math.pi,0]
    data_generator=DataGenerator(partial=True)
    occupancy_maps,ref_control_lists,adjacency_lists=data_generator.generate_map_control(self_pose_array,self_orientation_array)
    cv2.imshow("robot view (Synthesise)", occupancy_maps[0])
    cv2.waitKey(0)
    print(ref_control_lists)
    pass

