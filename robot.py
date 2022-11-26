"""
A robot template
author: Xinchi Huang
"""
from vrep.robot_executor_vrep import Executor
from vrep.robot_sensor_vrep import Sensor
# from realrobot.robot_executor_robomaster import Executor
# from realrobot.robot_sensor_realsense import Sensor
from controller import Controller


class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """

    def __init__(self):
        self.index = None
        self.GNN_model = None
        self.sensor_data = None
        self.control_data = None
        self.scene_data = None
        self.controller_type = "model"
        self.sensor = Sensor()
        self.executor = Executor(self.index)
        self.controller = Controller()
    def executor_initialize(self):
        self.executor.initialize()
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
        if self.controller_type == "expert":
            self.control_data = self.controller.centralized_control(
                self.index, self.sensor_data, self.scene_data
            )
        elif self.controller_type == "model":
            self.control_data = self.controller.decentralized_control(
                self.index, self.sensor_data, self.scene_data, number_of_agents=1
            )

        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """
        self.executor.execute_control(self.control_data)
