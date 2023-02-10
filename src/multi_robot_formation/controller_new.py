"""
A controller template

All controls are in robot coordinate
 y
 |
 |
 |
 r----------x

theta: counter clockwise relative to x-axis
"""


import collections
import math
import numpy as np
import torch
from model.GNN_based_model import (
    GnnMapBasic,
    GnnMapDecentralized,
    GnnPoseBasic,
    DummyModel,
)
import cv2
from utils.occupancy_map_simulator import MapSimulator
from comm_data import ControlData
class Controller:
    def __init__(self,desired_distance=1):
        self.desired_distance=desired_distance
        self.name=None

class CentralizedController(Controller):
    """
    A centralized controller
    """
    def __init__(self,desired_distance=1):
        super().__init__(desired_distance)
        self.name="Centralized"
    def get_control(self,index,neighbors,self_pose):
        """
        Return controls
        :param index: The robot index (type: int)
        :param neighbors: The neighbors of this robot (type: 2D iterator)
        :param self_position: Robot's self position (type: 1D iterator)
        :return: Control data (type: ControlData)
        """

        out_put = ControlData()

        if neighbors == None:
            print("No neighbor")
            out_put.robot_index = index
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put
        # if self_pose==None:
        #     print("self pose")
        #     out_put.robot_index = index
        #     out_put.velocity_x = 0
        #     out_put.velocity_y = 0
        #     return out_put


        self_x = self_pose[0]
        self_y = self_pose[1]
        theta=self_pose[2]
        velocity_sum_x = 0
        velocity_sum_y = 0

        for neighbor in neighbors:

            distance=((self_x - neighbor[1])**2+(self_y - neighbor[2])**2)**0.5
            rate = (distance - self.desired_distance) / distance
            velocity_x = rate * (self_x - neighbor[1])
            velocity_y = rate * (self_y - neighbor[2])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        ### transfrom global velocity to local velocity
        velocity_sum_x=velocity_sum_x*math.cos(theta)+velocity_sum_y*math.sin(theta)
        velocity_sum_y=-velocity_sum_x*math.sin(theta)+velocity_sum_y*math.cos(theta)

        out_put.robot_index = index
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y


        return out_put
class GnnMapBasicControllerSensor(Controller):
    def __init__(self,model_path,desired_distance=1.0,num_robot=5,input_height=100,input_width=100,use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param input_height: Occupancy map height (type: int)
        :param input_width: Occupancy map width (type: int)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name="GNN map Basic"
        self.model_path=model_path
        self.num_robot=num_robot

        self.input_height = input_height
        self.input_width = input_width

        self.use_cuda = use_cuda
        self.initialize_GNN_model()


    def initialize_GNN_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnMapBasic(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()
        
    def get_control(self,index,scene_data):
        """

        :param index: Robots' index (type: int)
        :param scene_data: Data from the scene (type: SceneData)
        :return:Control data (type: ControlData)
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
            (1, self.num_robot, self.input_width, self.input_height)
        )
        neighbor = np.zeros((self.num_robot, self.num_robot))
        ref = np.zeros((1, self.num_robot, 1))
        scale = np.zeros((1, self.num_robot, 1))
        for i in range(self.num_robot):
            input_occupancy_maps[0, i, :, :] = scene_data.observation_list[i].occupancy_map
            cv2.imshow("robot view "+str(i)+"(real)", input_occupancy_maps[0, i, :, :])
            cv2.waitKey(1)
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
        if self.use_cuda:
            input_tensor = input_tensor.to("cuda")
            neighbor = neighbor.to("cuda")
            ref = ref.to("cuda")
            scale = scale.to("cuda")
        self.GNN_model.eval()
        self.GNN_model.addGSO(neighbor)
        control = (
            self.GNN_model(input_tensor, ref, scale)[index].detach().cpu().numpy()
        )
        velocity_x = control[0][0]
        velocity_y = control[0][1]
        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y

        return out_put

class GnnMapBasicControllerSynthesise(Controller):
    def __init__(self, model_path, desired_distance=1.0, num_robot=5, input_height=100, input_width=100, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param input_height: Occupancy map height (type: int)
        :param input_width: Occupancy map width (type: int)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "GNN map Basic Synthesise"
        self.model_path = model_path
        self.num_robot = num_robot

        self.input_height = input_height
        self.input_width = input_width

        self.use_cuda = use_cuda
        self.initialize_GNN_model()
    def initialize_GNN_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnMapBasic(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def get_control(self, index, scene_data):
        """

        :param index: Robots' index (type: int)
        :param scene_data: Data from the scene (type: SceneData)
        :return:Control data (type: ControlData)
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
            (1, self.num_robot, self.input_width, self.input_height)
        )
        neighbor = np.zeros((self.num_robot, self.num_robot))
        ref = np.zeros((1, self.num_robot, 1))
        scale = np.zeros((1, self.num_robot, 1))
        position_lists_global = scene_data.position_list
        orientation_list = scene_data.orientation_list

        occupancy_map_simulator = MapSimulator(local=True)
        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            np.array(position_lists_global), np.array(orientation_list)
        )
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)
        for i in range(self.num_robot):
            occupancy_map_i = occupancy_maps[i]
            input_occupancy_maps[0, i, :, :] = occupancy_map_i
            cv2.imshow("robot view " + str(i) + "(Synthesise)", input_occupancy_maps[0, i, :, :])
            cv2.waitKey(1)
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
            self.GNN_model(input_tensor, ref, scale)[index].detach().cpu().numpy()
        )  ## model output
        velocity_x = control[0][0]
        velocity_y = control[0][1]

        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y

        return out_put

class GnnMapDecentralizedControllerSensor(Controller):
    def __init__(self, model_path, desired_distance=1.0, num_robot=5, input_height=100, input_width=100, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param input_height: Occupancy map height (type: int)
        :param input_width: Occupancy map width (type: int)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "GNN map Decentralized Sensor"
        self.model_path = model_path
        self.num_robot = num_robot

        self.input_height = input_height
        self.input_width = input_width

        self.use_cuda = use_cuda
        self.initialize_GNN_model()
    def initialize_GNN_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnMapDecentralized(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def get_control(self, index, scene_data):
        """

        :param index: Robots' index (type: int)
        :param scene_data: Data from the scene (type: SceneData)
        :return:Control data (type: ControlData)
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

        self_input_occupancy_maps = np.zeros(
            (1, self.num_robot, self.input_width, self.input_height)
        )
        neighbor = np.zeros((self.num_robot, self.num_robot))
        ref = np.zeros((1, self.num_robot, 1))
        scale = np.zeros((1, self.num_robot, 1))
        outer_msg = collections.defaultdict(int)
        for i in range(self.num_robot):
            if index == i:
                self_input_occupancy_maps[0, i, :, :] = scene_data.observation_list[i].occupancy_map
                cv2.imshow("robot view " + str(i) + "(real)", self_input_occupancy_maps[0, i, :, :])
                cv2.waitKey(1)
            ref[0, i, 0] = 0
            scale[0, i, 0] = self.desired_distance
        self_input_tensor = torch.from_numpy(self_input_occupancy_maps).double()

        for key, value in scene_data.adjacency_list.items():
            for n in value:
                neighbor[key][n[0]] = 1

        neighbor = torch.from_numpy(neighbor).double()
        neighbor = neighbor.unsqueeze(0)
        ref = torch.from_numpy(ref).double()
        scale = torch.from_numpy(scale).double()
        if self.use_cuda:
            self_input_tensor = self_input_tensor.to("cuda")
            neighbor = neighbor.to("cuda")
            ref = ref.to("cuda")
            scale = scale.to("cuda")
        self.GNN_model.eval()
        self.GNN_model.addGSO(neighbor)
        control = (
            self.GNN_model(self_input_tensor, outer_msg, index, ref, scale)[index].detach().cpu().numpy()
        )  ## model output
        velocity_x = control[0][0]
        velocity_y = control[0][1]
        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y
        return out_put

class GnnMapDecentralizedControllerSynthesise(Controller):
    def __init__(self, model_path, desired_distance=1.0, num_robot=5, input_height=100, input_width=100, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param input_height: Occupancy map height (type: int)
        :param input_width: Occupancy map width (type: int)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "GNN map Decentralized Synthesise"
        self.model_path = model_path
        self.num_robot = num_robot

        self.input_height = input_height
        self.input_width = input_width

        self.use_cuda = use_cuda
        self.initialize_GNN_model()
    def initialize_GNN_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnMapDecentralized(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def get_control(self, index, scene_data):
        """

        :param index: Robots' index (type: int)
        :param scene_data: Data from the scene (type: SceneData)
        :return:Control data (type: ControlData)
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

        self_input_occupancy_maps = np.zeros(
            (1, self.num_robot, self.input_width, self.input_height)
        )
        neighbor = np.zeros((self.num_robot, self.num_robot))
        ref = np.zeros((1, self.num_robot, 1))
        scale = np.zeros((1, self.num_robot, 1))

        position_lists_global = scene_data.position_list
        orientation_list = scene_data.orientation_list

        occupancy_map_simulator = MapSimulator(local=True)
        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            np.array(position_lists_global), np.array(orientation_list)
        )
        occupancy_maps = occupancy_map_simulator.generate_maps(position_lists_local)

        outer_msg = collections.defaultdict(int)

        for i in range(self.num_robot):

            occupancy_map_i = occupancy_maps[i]
            if index == i:
                self_input_occupancy_maps[0, i, :, :] = occupancy_map_i
                cv2.imshow("robot view " + str(i) + "(Synthesise)", self_input_occupancy_maps[0, i, :, :])
                cv2.waitKey(1)
            ref[0, i, 0] = 0
            scale[0, i, 0] = self.desired_distance

        self_input_tensor = torch.from_numpy(self_input_occupancy_maps).double()

        for key, value in scene_data.adjacency_list.items():
            for n in value:
                neighbor[key][n[0]] = 1

        neighbor = torch.from_numpy(neighbor).double()
        neighbor = neighbor.unsqueeze(0)
        ref = torch.from_numpy(ref).double()
        scale = torch.from_numpy(scale).double()
        if self.use_cuda:
            self_input_tensor = self_input_tensor.to("cuda")
            neighbor = neighbor.to("cuda")
            ref = ref.to("cuda")
            scale = scale.to("cuda")
        self.GNN_model.eval()
        self.GNN_model.addGSO(neighbor)
        control = (
            self.GNN_model(self_input_tensor, outer_msg, index, ref, scale)[index].detach().cpu().numpy()
        )
        velocity_x = control[0][0]
        velocity_y = control[0][1]


        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y

        return out_put

class GnnPoseBasicController(Controller):
    def __init__(self, model_path, desired_distance=1.0, num_robot=5, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "GNN Pose"
        self.model_path = model_path
        self.num_robot = num_robot

        self.use_cuda = use_cuda
        self.initialize_model()
    def initialize_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnPoseBasic(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def get_control(self, index, scene_data):
        """

        :param index: Robots' index (type: int)
        :param scene_data: Data from the scene (type: SceneData)
        :return:Control data (type: ControlData)
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

        neighbor = np.zeros((self.num_robot, self.num_robot))
        ref = np.zeros((1, self.num_robot, 1))
        scale = np.zeros((1, self.num_robot, 1))

        position_lists_global = scene_data.position_list
        orientation_list = scene_data.orientation_list
        occupancy_map_simulator = MapSimulator(local=True)
        (
            position_lists_local,
            self_orientation,
        ) = occupancy_map_simulator.global_to_local(
            position_lists_global, orientation_list
        )
        position_array_local = np.zeros((1, 5, 4, 3))
        try:
            for i in range(len(position_lists_local[0])):
                for j in range(len(position_lists_local[0][i])):
                    position_array_local[0][i][j] = position_lists_local[i][j]
        except:
            return out_put


        input_tensor = torch.from_numpy(position_array_local).double()

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
            self.GNN_model(input_tensor, ref, scale)[index].cpu().detach().numpy()
        )
        velocity_x = control[0][0]
        velocity_y = control[0][1]

        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y

        return out_put


class DummyController(Controller):
    def __init__(self, model_path, desired_distance=1.0, num_robot=5, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "Dummy"
        self.model_path = model_path
        self.num_robot = num_robot

        self.use_cuda = use_cuda

    def initialize_GNN_model(self):
        """
        Initialize GNN model
        """
        self.GNN_model = GnnPoseBasic(number_of_agent=self.num_robot, use_cuda=True)
        if not self.use_cuda:
            self.GNN_model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.GNN_model.load_state_dict(torch.load(self.model_path))
            self.GNN_model.to("cuda")
        self.GNN_model.eval()

    def get_control(self, index,position_lists_local):
        """

        :param index: Robots' index (type: int)
        :param position_lists_local: Data from the scene (type: 2D iterator)
        :return:Control data (type: ControlData)
        """
        print(position_lists_local)
        out_put = ControlData()
        position_array_local = np.zeros((1, 1, len(position_lists_local), 3))
        for i in range(len(position_lists_local)):
            position_array_local[0][0][i] = position_lists_local[i]

        input_tensor = torch.from_numpy(position_array_local).double()

        # print("TENSOR")

        if self.use_cuda:
            input_tensor = input_tensor.to("cuda")
        self.GNN_model.eval()

        control = (
            self.GNN_model(input_tensor)[index].detach().cpu().numpy()
        )  ## model output
        velocity_x = control[0][0]
        velocity_y = control[0][1]

        theta = 0

        out_put.robot_index = index
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y

        return out_put


