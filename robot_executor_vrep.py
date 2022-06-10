"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""
import vrep_interface


class Executor:
    """
    A class to execute control from controller
    """

    def __init__(self):
        self.client_id = None
        self.robot_handle = None
        self.motor_left_handle = None
        self.motor_right_handle = None
        self.point_cloud_handle = None

    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in simulator/real world
        :param control_data: Controls to be execute
        """
        omega_left = control_data.omega_left
        omega_right = control_data.omega_right
        vrep_interface.post_control(
            self.client_id,
            self.motor_left_handle,
            self.motor_right_handle,
            omega_left,
            omega_right,
        )
        vrep_interface.synchronize(self.client_id)
