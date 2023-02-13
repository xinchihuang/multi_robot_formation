"""
A robot template
author: Xinchi Huang
"""
# from vrep.robot_executor_vrep import Executor
# from vrep.robot_sensor_vrep import Sensor

# from .realrobot.robot_executor_robomaster import Executor
# from .realrobot.robot_sensor_realsense import Sensor
import os

# from .controller import Controller
from src.multi_robot_formation.controller_new import *
from src.multi_robot_formation.comm_data import ControlData,SceneData,SensorData



class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """

    def __init__(
        self, sensor, executor,model_path="saved_model/model_12000.pth", platform="vrep", controller_type="expert",sensor_type="synthesise"
    ):
        self.index = None
        self.GNN_model = None
        self.sensor_data = SensorData()
        self.control_data = ControlData()
        self.scene_data = SceneData()

        self.platform = platform
        self.controller_type = controller_type
        self.sensor_type=sensor_type
        self.sensor = sensor
        self.executor = executor

        self.model_path=model_path

        if self.controller_type == "expert":
            self.controller=CentralizedController()
        elif self.controller_type == "model_basic":
            if self.sensor_type == "real":
                self.controller=GnnMapBasicControllerSensor(self.model_path)
            if self.sensor_type == "synthesise":
                self.controller = GnnMapBasicControllerSynthesise(self.model_path)
        elif self.controller_type == "model_decentralized":
            if self.sensor_type == "real":
                self.controller=GnnMapDecentralizedControllerSensor(self.model_path)
            if self.sensor_type == "synthesise":
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

    def get_control_data(self):
        """
        Get controls
        :return: Control data
        """
        if self.controller_type == "expert":
            if not self.scene_data==None and not self.sensor_data==None and not self.scene_data.adjacency_list==None:
                self.control_data=self.controller.get_control(self.index,self.scene_data.adjacency_list[self.index],self.sensor_data.position)
        elif self.controller_type == "model_basic":
            if self.sensor_type == "real":
                self.control_data=self.controller.get_control(self.index,self.scene_data)
            if self.sensor_type == "synthesise":
                self.control_data=self.controller.get_control(self.index,self.scene_data)
        elif self.controller_type == "model_decentralized":
            if self.sensor_type == "real":
                self.control_data = self.controller.get_control(self.index, self.scene_data)
            if self.sensor_type == "synthesise":
                self.control_data=self.controller.get_control(self.index,self.scene_data)
        elif self.controller_type == "model_dummy":
            self.control_data=self.controller.get_control(self.index,self.scene_data)

        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """

        if self.platform == "vrep":
            if self.sensor_data==None:
                self.control_data.orientation=[0,0,0]
            else:

                self.control_data.orientation = self.sensor_data.orientation
        self.executor.execute_control(self.control_data)
