"""
A robot template
author: Xinchi Huang
"""
from robot_executor_vrep import Executor, ControlData
from robot_sensor_vrep import Sensor


class Robot:
    """
    A robot template. Used for handling different components and store data for components.
    """
    def __init__(self):
        self.max_velocity = 1.2
        self.sensor_data = None
        self.control_data = None
        self.sensor = Sensor()
        self.executor = Executor()

    def get_sensor_data(self):
        """
        Read sensor data
        :return: Sensor data
        """
        self.sensor_data = self.sensor.get_sensor_data()
        return self.sensor_data

    def get_control_data(self):
        """
        Get controls
        :return: Control data
        """
        robot_control_data = ControlData()
        robot_control_data.omega_left = 10
        robot_control_data.omega_right = 10
        self.control_data = robot_control_data
        return self.control_data

    def execute_control(self):
        """
        Execute control from controller
        :return:
        """
        self.executor.execute_control(self.control_data)
