"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""
from vrep import vrep_interface


class Executor:
    """
    A class to execute control from controller
    """

    def __init__(self,index):
        self.robot_index = index
        self.client_id = None
        self.robot_handle = None
        self.motor_left_handle = None
        self.motor_right_handle = None
        self.point_cloud_handle = None
    def initialize(self):
        (
            robot_handle,
            motor_left_handle,
            motor_right_handle,
            point_cloud_handle,
        ) = vrep_interface.get_vrep_handle(self.client_id, self.robot_index)
        self.robot_handle = robot_handle
        self.motor_left_handle = motor_left_handle
        self.motor_right_handle = motor_right_handle
        self.point_cloud_handle = point_cloud_handle
    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in simulator/real world
        :param control_data: Controls to be execute
        """
        omega_left = control_data.omega_left
        omega_right = control_data.omega_right
        print("index", control_data.robot_index)
        print("left", omega_left)
        print("right", omega_right)

        vrep_interface.post_control(
            self.client_id,
            self.motor_left_handle,
            self.motor_right_handle,
            omega_left,
            omega_right,
        )
        # vrep_interface.synchronize(self.client_id)
