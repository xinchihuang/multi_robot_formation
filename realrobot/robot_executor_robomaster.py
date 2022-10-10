"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""



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
        Use interface/APIs to execute control in real world
        :param control_data: Controls to be execute
        """
        omega_left = control_data.omega_left
        omega_right = control_data.omega_right
        print("index", control_data.robot_index)
        print("left", omega_left)
        print("right", omega_right)

