"""
A robot template
author: Xinchi Huang
"""
from robot_executor_vrep import executor, control_data
from robot_sensor_vrep import sensor, sensor_data


class robot:
    def __init__(self):
        self.max_velocity = 1.2
        self.sensor_data = None
        self.control_data = None
        self.sensor = sensor()
        self.executor = executor()

    def get_sensor_data(self):
        self.sensor_data = self.sensor.get_sensor_data()
        return self.sensor_data

    def get_control_data(self):
        robot_control_data = control_data()
        robot_control_data.omega_left = 10
        robot_control_data.omega_right = 10
        self.control_data = robot_control_data
        return self.control_data

    def execute_control(self):
        self.executor.execute_control(self.control_data)
