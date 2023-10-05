import math
import numpy as np
import torch
from utils.gabreil_graph import get_gabreil_graph_local,global_to_local
from comm_data import ControlData
from model.vit_model import ViT
class BaseController:
    def get_data(self,data):
        pass
    def get_control(self):
        pass
class CentralizedController(BaseController):
    """
    A centralized controller
    """
    def __init__(self,desired_distance=2):
        self.name="CentralizedController"
        self.desired_distance=desired_distance
    def get_data(self,data):
        super().get_data(data)
        self.robot_id = data["robot_id"]
        self.neighbors = data["neighbors"]
        self.self_pose = data["self_pose"]
    def get_control(self):
        print(self.name)
        out_put = ControlData()

        if self.neighbors == None:
            print("No neighbor")
            out_put.robot_index = self.robot_id
            out_put.velocity_x = 0
            out_put.velocity_y = 0
            return out_put
        # if self_pose==None:
        #     print("self pose")
        #     out_put.robot_index = index
        #     out_put.velocity_x = 0
        #     out_put.velocity_y = 0
        #     return out_put
        self_x = self.self_pose[0]
        self_y = self.self_pose[1]
        theta=self.self_pose[2]
        velocity_sum_x = 0
        velocity_sum_y = 0
        for neighbor in self.neighbors:

            distance=((self_x - neighbor[1])**2+(self_y - neighbor[2])**2)**0.5
            rate = (distance-0.5 - self.desired_distance) / (distance-0.5)
            velocity_x = rate * (self_x - neighbor[1])
            velocity_y = rate * (self_y - neighbor[2])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        ### transfrom global velocity to local velocity
        velocity_sum_x=velocity_sum_x*math.cos(theta)+velocity_sum_y*math.sin(theta)
        velocity_sum_y=-velocity_sum_x*math.sin(theta)+velocity_sum_y*math.cos(theta)

        out_put.robot_index = self.robot_id
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y


        return out_put
class LocalExpertController(BaseController):
    def __init__(self,desired_distance=2,sensor_range=5,sensor_angle=math.pi/2,safe_margin=0.4,K_f=1,K_m=1,K_omega=1,max_speed=1,max_omega=1):
        self.name = "LocalExpertController"
        self.desired_distance = desired_distance
        self.sensor_range = sensor_range
        self.sensor_angle = sensor_angle
        self.safe_margin = safe_margin
        self.K_f = K_f
        self.K_m = K_m
        self.K_omega = K_omega
        self.max_speed=max_speed
        self.max_omega=max_omega
    def get_data(self,data):
        super().get_data(data)
        self.robot_id = data["robot_id"]
        self.pose_list = data["pose_list"]

    def get_control(self):
        """
        :param position_list: global position list for training
        """
        super().get_control()
        out_put = ControlData()
        desired_distance = self.desired_distance
        gabreil_graph_local = get_gabreil_graph_local(self.pose_list, self.sensor_range, self.sensor_angle)
        pose_array_local= (self.pose_list)
        neighbor_list = gabreil_graph_local[self.robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        velocity_sum_omega = 0
        for neighbor_id in range(len(neighbor_list)):
            if neighbor_id == self.robot_id or neighbor_list[neighbor_id] == 0:
                continue
            position_local=pose_array_local[self.robot_id][neighbor_id]
            distance_formation = (position_local[0] ** 2 + position_local[1] ** 2) ** 0.5
            rate_f = (distance_formation - desired_distance) / distance_formation
            velocity_x_f = rate_f * position_local[0]
            velocity_y_f = rate_f * position_local[1]

            velocity_omega = math.atan2(position_local[1],(position_local[0]))
            # print(robot_id,neighbor_id,position_local[1],position_local[0],velocity_omega)

            gamma = math.atan2(position_local[1], (position_local[0]))
            distance_left = position_local[0] * math.sin(self.sensor_angle / 2) + position_local[1] * math.cos(
                self.sensor_angle / 2)
            if distance_left > self.safe_margin:
                rate_l=0
            else:
                # print(robot_id,neighbor_id,distance_left)
                rate_l = (self.safe_margin - distance_left) / distance_left
            velocity_x_l = rate_l * position_local[0]*(-math.sin(self.sensor_angle/2))
            velocity_y_l = rate_l * position_local[1] * (-math.cos(self.sensor_angle / 2))

            distance_right = position_local[0] * math.sin(self.sensor_angle / 2) - position_local[1] * math.cos(
                self.sensor_angle / 2)
            if distance_right > self.safe_margin:
                rate_r=0
            else:
                # print(robot_id, neighbor_id, distance_right)
                rate_r = (self.safe_margin - distance_right) / distance_right
            velocity_x_r = rate_r * position_local[0] * (-math.sin(self.sensor_angle / 2))
            velocity_y_r = rate_r * position_local[1] * (math.cos(self.sensor_angle / 2))

            distance_sensor = self.sensor_range - ((position_local[0]) ** 2 + (position_local[1])**2 )** 0.5
            if distance_sensor > self.safe_margin:
                rate_s=0
            else:
                # print(robot_id, neighbor_id, distance_sensor)
                rate_s = (self.safe_margin - distance_sensor) / distance_sensor
            velocity_x_s = rate_s * position_local[0] * (math.cos(gamma))
            velocity_y_s = rate_s * position_local[1] * (math.sin(gamma))

            velocity_sum_x += self.K_f*velocity_x_f+self.K_m*(velocity_x_l+velocity_x_r+velocity_x_s)
            velocity_sum_y += self.K_f*velocity_y_f+self.K_m*(velocity_y_l+velocity_y_r+velocity_y_s)
            velocity_sum_omega += self.K_omega*velocity_omega
        # print(robot_id,velocity_x_f,velocity_x_l,velocity_x_r,velocity_x_s)
        out_put.velocity_x=velocity_sum_x if abs(velocity_sum_x)<self.max_speed else self.max_speed*abs(velocity_sum_x)/velocity_sum_x
        out_put.velocity_y=velocity_sum_y if abs(velocity_sum_y)<self.max_speed else self.max_speed*abs(velocity_sum_y)/velocity_sum_y
        out_put.omega=velocity_sum_omega if abs(velocity_sum_omega)<self.max_omega else self.max_omega*abs(velocity_sum_omega)/velocity_sum_omega
        return

class VitController(BaseController):
    def __init__(self, model_path, desired_distance=2.0, num_robot=5, input_height=100, input_width=100, use_cuda=True):
        """
        :param desired_distance: Desired formation distance (type: float)
        :param num_robot: The number of robots (type: int)
        :param model_path: Path to pretrained model (type: string)
        :param input_height: Occupancy map height (type: int)
        :param input_width: Occupancy map width (type: int)
        :param use_cuda: Decide whether to use cuda (type: bool)
        """
        super().__init__(desired_distance)
        self.name = "VitController"
        self.model_path = model_path
        self.num_robot = num_robot
        self.input_height = input_height
        self.input_width = input_width

        self.use_cuda = use_cuda
        self.initialize_model()
    def initialize_model(self):
        """
        Initialize ViT model
        """
        # print(self.name)
        self.model = ViT(
        image_size = 100,
        patch_size = 10,
        num_classes = 3,
        dim = 256,
        depth = 3,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    ).double()
        if not self.use_cuda:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        else:
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.to("cuda")
        self.model.eval()
    def get_data(self,data):
        self.robot_id=data["robot_id"]
        self.occupancy_map= data["occupancy_map"]
    def get_control(self):
        """

        :param index: Robots' index (type: int)
        :param occupancy_map:
        :return:Control data (type: ControlData)
        """
        out_put = ControlData()
        self_input_occupancy_maps = np.zeros(
            (1, 1, self.input_width, self.input_height)
        )

        self_input_occupancy_maps[0, 0, :, :] = self.occupancy_map
        # cv2.imshow("robot view " + str(index) + "(Synthesise)", occupancy_map)
        # cv2.waitKey(1)
        self_input_tensor = torch.from_numpy(self_input_occupancy_maps).double()

        if self.use_cuda:
            self_input_tensor = self_input_tensor.to("cuda")
        self.model.eval()


        control = (
            self.model(self_input_tensor,task="control").detach().cpu().numpy()
        )
        print(self.robot_id,control)
        velocity_x = control[0][0]
        velocity_y = control[0][1]
        omega=control[0][2]
        out_put.robot_index = self.robot_id
        out_put.velocity_x = velocity_x
        out_put.velocity_y = velocity_y
        out_put.omega=omega

        return out_put

