"""
A robot template
author: Xinchi Huang
"""
# from vrep.robot_executor_vrep import Executor
# from vrep.robot_sensor_vrep import Sensor

# from .realrobot.robot_executor_robomaster import Executor
# from .realrobot.robot_sensor_realsense import Sensor
import os

print(os.getcwd())
# from .controller import Controller
from controller_new import *



class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """

    def __init__(
        self, sensor, executor,model_path="saved_model/model_map_local_partial.pth", platform="robomaster", controller_type="model_decentralized",sensor_type="real"
    ):
        self.index = None
        self.GNN_model = None
        self.sensor_data = None
        self.control_data = None
        self.scene_data = None

        self.platform = platform
        self.controller_type = controller_type
        self.sensor = sensor
        self.executor = executor

        self.model_path=model_path

        if self.controller_type == "expert":
            self.controller=CentralizedController()
        elif self.controller_type == "model_basic":
            if sensor_type == "real":
                self.controller=GnnMapBasicControllerSensor(self.model_path)
            if sensor_type == "synthesise":
                self.controller = GnnMapBasicControllerSynthesise(self.model_path)
        elif self.controller_type == "model_decentralized":
            if sensor_type == "real":
                self.controller=GnnMapDecentralizedControllerSensor(self.model_path)
            if sensor_type == "synthesise":
                self.controller = GnnMapDecentralizedControllerSynthesise(self.model_path)
        elif self.controller_type == "model_dummy":
            self.controller=DummyController(self.model_path)



    def get_sensor_data(self):

        """
        Read sensor data from sensor in simulator/realworld
        :return: Sensor data
        """
        self.sensor_data = self.sensor.get_sensor_data()
        return self.sensor_data

    def get_control_data_old(self):
        """
        Get controls
        :return: Control data
        """
        if self.controller_type == "expert":
            model_data = self.controller.centralized_control(
                self.index, self.sensor_data, self.scene_data
            )
        elif self.controller_type == "model":
            # model_data = self.controller.decentralized_control(
            #     self.index, self.sensor_data, self.scene_data, number_of_agents=5
            # )
            model_data = self.controller.decentralized_control_real(
                self.index, self.sensor_data, self.scene_data, number_of_agents=5
            )
            # mode
        elif self.controller_type == "model_pose":
            model_data = self.controller.decentralized_control_pose(
                self.index, self.sensor_data, self.scene_data, number_of_agents=5
            )
        elif self.controller_type == "model_dummy":
            # model_data = self.controller.decentralized_control_dummy(
            #     self.index, self.sensor_data, self.scene_data, number_of_agents=3
            # )
            model_data = self.controller.decentralized_control_dummy_real(
                self.index, self.sensor_data
            )
            # print("robot ", self.index)
            # if not self.scene_data==None:
            #     print("position list", self.scene_data.position_list)
            #     print("orientation list", self.scene_data.orientation_list)
            # print("expert ", expert_data.velocity_x, expert_data.velocity_y)
        print(self.controller_type, model_data.velocity_x, model_data.velocity_y)
        self.control_data = model_data

        return self.control_data

    def get_control_data(self):
        """
        Get controls
        :return: Control data
        """

        self.control_data=self.controller.get_control(self.index,self.scene_data)
        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """

        if self.platform == "vrep":
            self.control_data.orientation = self.sensor_data.orientation
        self.executor.execute_control(self.control_data)
