"""
A sensor template. Get information from simulator/real-world
author: Xinchi Huang
"""
import vrep_interface


class SensorData:
    """
    A class for record sensor data
    """

    def __init__(self):
        self.robot_index = None
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.velodyne_points = None


class Sensor:
    """
    Robot sensor
    """

    def __init__(self):
        self.client_id = None
        self.robot_handle = None
        self.robot_index = None

    def get_sensor_data(self):
        """
        Get sensor readings
        :return: Data from sensor and simulator
        """
        robot_sensor_data = SensorData()
        position, orientation = vrep_interface.get_robot_pose(
            self.client_id, self.robot_handle
        )
        (
            linear_velocity,
            angular_velocity,
            velodyne_points,
        ) = vrep_interface.get_sensor_data(
            self.client_id, self.robot_handle, self.robot_index
        )
        robot_sensor_data.robot_index = self.robot_index
        robot_sensor_data.position = position
        robot_sensor_data.orientation = orientation
        robot_sensor_data.linear_velocity = linear_velocity
        robot_sensor_data.angular_velocity = angular_velocity
        robot_sensor_data.velodyne_points = velodyne_points
        return robot_sensor_data
