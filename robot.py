"""
A robot template
author: Xinchi Huang
"""
from vrep.robot_executor_vrep import Executor
from vrep.robot_sensor_vrep import Sensor
from controller import Controller
from model.GNN_based_model import DecentralController

class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """

    def __init__(self):
        self.index = None
        self.GNN_model=None
        self.sensor_data = None
        self.control_data = None
        self.scene_data = None
        self.sensor = Sensor()
        self.executor = Executor()
        self.controller = Controller()

    # def set_up_components(self):
    #     """
    #
    #     :return:
    #     """
    #     pass


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
        # self.control_data = self.controller.centralized_control(
        #     self.index, self.sensor_data, self.scene_data
        # )

        self.control_data = self.controller.decentralized_control(
            self.index, self.sensor_data, self.scene_data,self.GNN_model,number_of_agents=5
        )

        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """
        self.executor.execute_control(self.control_data)
