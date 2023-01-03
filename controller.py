"""
A controller template
"""
import collections
import math
import numpy as np
import torch
from model.GNN_based_model import DecentralController
import cv2
from utils.occupancy_map_simulator import MapSimulator

from comm_data import ControlData
# class ControlData:
#     """
#     A data structure for passing control signals to executor
#     """
#
#     def __init__(self):
#         self.robot_index = None
#         # self.omega_left = 0
#         # self.omega_right = 0
#
#         self.velocity_x = 0
#         self.velocity_y = 0


class Controller:
    """
    A template for robot controller. Support centralized and decentralized control
    """

    def __init__(self):
        """
        desired_distance: Desired formation distance
        centralized_k: For centralized control, a manually defined rate for total velocity
        max_velocity: Maximum linear velocity
        wheel_adjustment: Used for transform linear velocity to angular velocity
        """
        self.desired_distance = 2.0
        self.centralized_k = 1
        self.max_velocity = 1.2
        self.wheel_adjustment = 10.25
        self.GNN_model = None
        self.use_cuda = False

    def centralized_control(self, index, sensor_data, scene_data):
        """
        A centralized control, Expert control
        :param index: Robot index
        :param sensor_data: Data from robot sensor
        :param scene_data: Data from the scene
        :return: Control data
        """
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
        velocity_sum_x = 0
        velocity_sum_y = 0
        for neighbor in neighbors:
            rate = (neighbor[3] - self.desired_distance) / neighbor[3]
            velocity_x = rate * (self_x - neighbor[1])
            velocity_y = rate * (self_y - neighbor[2])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        # transform speed to wheels speed
        out_put.robot_index = index
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y

        return out_put

    def initialize_GNN_model(self, num_robot, model_path):
        """

        :param num_robot: The number of robots in simulation run
        :param model_path: Path to pretrained model
        :return:
        """
        self.GNN_model = DecentralController(number_of_agent=num_robot, use_cuda=False)
        self.GNN_model.load_state_dict(torch.load(model_path))
        if self.use_cuda:
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def centralized_control_line(
        self,
        index,
        sensor_data,
        scene_data,
        separation=2,
        angle=math.pi / 4,
        max_distance=10,
        K_c=2,
    ):
        """
        A centralized controller to form a line, Expert control
        :param index: Robot index
        :param sensor_data: Sensor_data from robots' sensors or simulators
        :param scene_data: Data from the scene
        :param separation: Distance between each robots
        :param angle: Desired line angle to the positive part of x axis
        :param max_distance: Robots only interact within this region
        :return: Control data
        """
        out_put = ControlData()

        if not scene_data:
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put
        target_position = []
        robot_num = len(scene_data)
        for i in range(robot_num):
            target_position.append(
                [i * separation * math.cos(angle), i * separation * math.sin(angle)]
            )
        robot_position_dict = collections.defaultdict(tuple)
        for _, value in scene_data.adjacency_list.items():
            for item in value:
                robot_position_dict[item[0]] = (item[1], item[2])
        neighbor_list = []
        desired_force = []
        for ri in range(robot_num):
            if (
                not ri == index
                and (
                    (robot_position_dict[index][0] - robot_position_dict[ri][0]) ** 2
                    + (robot_position_dict[index][1] - robot_position_dict[ri][1]) ** 2
                )
                ** 0.5
                < max_distance
            ):
                neighbor_list.append(
                    (robot_position_dict[ri][0], robot_position_dict[ri][1])
                )
                desired_force.append(
                    (
                        target_position[index][0] - target_position[ri][0],
                        target_position[index][1] - target_position[ri][1],
                    )
                )
        #### get speed
        velocity_index_x = 0
        velocity_index_y = 0
        self_position = sensor_data.position[:2]
        theta = sensor_data.orientation[2]
        neighbor_num = len(neighbor_list)
        for i in range(neighbor_num):
            w_ij = [
                self_position[0] - neighbor_list[i][0] - desired_force[i][0],
                self_position[1] - neighbor_list[i][1] - desired_force[i][1],
            ]
            velocity_index_x = velocity_index_x - K_c * w_ij[0]
            velocity_index_y = velocity_index_y - K_c * w_ij[1]

        self_robot_index = sensor_data.robot_index
        out_put.robot_index = self_robot_index
        out_put.velocity_x = velocity_index_x
        out_put.velocity_y = velocity_index_y
        return out_put

    def decentralized_control(
        self,
        index,
        sensor_data,
        scene_data,
        number_of_agents=5,
        input_height=100,
        input_width=100,
        local=True
    ):
        """

        :param index: Robots' index
        :param sensor_data: Sensor_data from robots' sensors or simulators
        :param scene_data: Data from the scene
        :param number_of_agents: The number of agent
        :param input_height: Input occupancy map height
        :param input_width: Input occupancy map width
        :return:
        """

        out_put = ControlData()
        if not scene_data:

            print("No scene data")
            out_put.robot_index = index
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put
        if not scene_data.observation_list:
            print("No observation")
            out_put.robot_index = index
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put
        if not scene_data.position_list or not scene_data.orientation_list:
            print("No data")
            out_put.robot_index = index
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put

        input_occupancy_maps = np.zeros(
            (1, number_of_agents, input_width, input_height)
        )
        neighbor = np.zeros((number_of_agents, number_of_agents))
        ### need to be removed later
        ref = np.zeros((1, number_of_agents, 1))
        ### control the formation distance
        scale = np.zeros((1, number_of_agents, 1))
        position_lists_global = scene_data.position_list
        # print("robot index",index)
        # print(position_lists_global[index])
        orientation_list = scene_data.orientation_list

        occupancy_map_simulator=MapSimulator(rotate=local)
        position_lists_local, self_pose = occupancy_map_simulator.global_to_local(np.array(position_lists_global),np.array(orientation_list))
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)

        for i in range(number_of_agents):
            # print(sensor_data.occupancy_map)
            ### need to be modified
            occupancy_map_i = occupancy_maps[i]
            cv2.imshow(str(i), occupancy_map_i)
            cv2.waitKey(1)
            input_occupancy_maps[0, i, :, :] = occupancy_map_i
            ref[0, i, 0] = 0
            scale[0, i, 0] = self.desired_distance

        input_tensor = torch.from_numpy(input_occupancy_maps).double()

        for key, value in scene_data.adjacency_list.items():
            for n in value:
                neighbor[key][n[0]] = 1

        neighbor = torch.from_numpy(neighbor).double()
        neighbor = neighbor.unsqueeze(0)
        ref = torch.from_numpy(ref).double()
        scale = torch.from_numpy(scale).double()
        # print("TENSOR")

        if self.use_cuda:
            input_tensor = input_tensor.to("cuda")
            neighbor = neighbor.to("cuda")
            ref = ref.to("cuda")
            scale = scale.to("cuda")
        self.GNN_model.eval()
        self.GNN_model.addGSO(neighbor)

        control = (
            self.GNN_model(input_tensor, ref, scale)[index].detach().numpy()
        )  ## model output
        velocity_x=control[0][0]
        velocity_y=control[0][1]

        if local:
            theta=sensor_data.orientation[2]
            velocity_x_global=velocity_x*math.sin(theta)+velocity_y*math.cos(theta)
            velocity_y_global=-velocity_x*math.cos(theta)+velocity_y*math.sin(theta)
            velocity_x=velocity_x_global
            velocity_y=velocity_y_global

        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y



        return out_put
