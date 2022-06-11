"""
A robot template
author: Xinchi Huang
"""
from robot_executor_vrep import Executor
from robot_sensor_vrep import Sensor
from controller import Controller


class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """

    def __init__(self):
        self.index = None
        self.sensor_data = None
        self.control_data = None
        self.network_data = None
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
        self.control_data = self.controller.centralized_control(
            self.index, self.sensor_data, self.network_data
        )
        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """
        self.executor.execute_control(self.control_data)
