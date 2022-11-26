"""
A controller template
"""
import collections
import math
import numpy as np
import torch
from model.GNN_based_model import DecentralController


class ControlData:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        self.omega_left = 0
        self.omega_right = 0


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

    def velocity_transform(self, velocity_x, velocity_y, theta):
        """
        Transform robot velocity to wheels velocity
        :param velocity_x:  robot velocity x (float)
        :param velocity_y: robot velocity y (float)
        :param theta: Robot orientation
        :return: wheel velocity left and right (float)
        """
        kk = self.centralized_k
        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        wheel_velocity_left = M11 * velocity_x + M12 * velocity_y
        wheel_velocity_right = M21 * velocity_x + M22 * velocity_y

        if (
            math.fabs(wheel_velocity_right) >= math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_right) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_right)
        elif (
            math.fabs(wheel_velocity_right) < math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_left) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_left)
        else:
            alpha = 1

        wheel_velocity_left = alpha * wheel_velocity_left
        wheel_velocity_right = alpha * wheel_velocity_right
        return wheel_velocity_left, wheel_velocity_right

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
            out_put.omega_left = 0
            out_put.omega_right = 0
            return out_put
        # self_robot_index = index

        self_position = sensor_data.position
        # self_orientation = sensor_data.orientation
        self_x = self_position[0]
        self_y = self_position[1]
        neighbors = scene_data.adjacency_list[index]
        # print(neighbors)
        velocity_sum_x = 0
        velocity_sum_y = 0
        for neighbor in neighbors:
            rate = (neighbor[3] - self.desired_distance) / neighbor[3]
            velocity_x = rate * (self_x - neighbor[1])
            velocity_y = rate * (self_y - neighbor[2])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        # transform speed to wheels speed
        theta = sensor_data.orientation[2]
        wheel_velocity_left, wheel_velocity_right = self.velocity_transform(
            velocity_sum_x, velocity_sum_y, theta
        )
        out_put.robot_index = index
        out_put.omega_left = wheel_velocity_left * self.wheel_adjustment
        out_put.omega_right = wheel_velocity_right * self.wheel_adjustment
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
            out_put.omega_left = 0
            out_put.omega_right = 0
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
        wheel_velocity_left, wheel_velocity_right = self.velocity_transform(
            velocity_index_x, velocity_index_y, theta
        )

        self_robot_index = sensor_data.robot_index
        out_put.robot_index = self_robot_index
        out_put.omega_left = wheel_velocity_left * self.wheel_adjustment
        out_put.omega_right = wheel_velocity_right * self.wheel_adjustment
        return out_put

    def decentralized_control(
        self,
        index,
        sensor_data,
        scene_data,
        number_of_agents=3,
        input_height=100,
        input_width=100,
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
            out_put.omega_left = 0
            out_put.omega_right = 0
            return out_put
        if not scene_data.observation_list:
            print("No observation")
            out_put.omega_left = 0
            out_put.omega_right = 0
            return out_put
        # if type(sensor_data.occupancy_map)==None:
        #     print("No occupancy_map")
        #     out_put.omega_left = 0
        #     out_put.omega_right = 0
        #     return out_put
        input_occupancy_maps = np.zeros(
            (1, number_of_agents, input_width, input_height)
        )
        neighbor = np.zeros((number_of_agents, number_of_agents))
        ### need to be removed later
        ref = np.zeros((1, number_of_agents, 1))
        ### control the formation distance
        scale = np.zeros((1, number_of_agents, 1))
        for i in range(number_of_agents):
            # print(sensor_data.occupancy_map)
            ### need to be modified
            input_occupancy_maps[0, i, :, :] = scene_data.observation_list[i].occupancy_map
            ref[0, i, 0] = 0
            scale[0, i, 0] = self.desired_distance
        ### a
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

        #### Set a threshold to eliminate small movements
        # threshold=0.05
        control = self.GNN_model(input_tensor, ref, scale)[index]  ## model output

        # torch.where(control<threshold, 0., control)
        # torch.where(control>-threshold, 0., control)

        out_put.robot_index = index
        out_put.omega_left = float(control[0][0]) * self.wheel_adjustment
        out_put.omega_right = float(control[0][1]) * self.wheel_adjustment

        # out_put = [control]
        # print("Control",out_put)
        # # print(outs)
        return out_put
