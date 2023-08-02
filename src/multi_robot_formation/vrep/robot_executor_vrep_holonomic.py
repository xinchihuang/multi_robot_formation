"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""
from vrep import vrep_interface_holonomic
import math


class ExecutorHolonomic:
    """
    A class to execute control from controller
    centralized_k: For centralized control, a manually defined rate for total velocity
    max_velocity: Maximum linear velocity
    wheel_adjustment: Used for transform linear velocity to angular velocity
    """

    def __init__(self):
        self.robot_index = None
        self.client_id = None
        self.robot_handle = None
        self.omniRob_FL_handle = None
        self.omniRob_FR_handle = None
        self.omniRob_RL_handle = None
        self.omniRob_RR_handle = None
        self.point_cloud_handle = None

        self.centralized_k = 1
        self.max_velocity = 0.5
        self.wheel_adjustment = 0.421

        self.rotate_direction=1

    def velocity_transform(self, velocity_x, velocity_y, theta,omega):
        """
        Transform robot velocity to wheels velocity
        :param velocity_x:  robot velocity x (float)
        :param velocity_y: robot velocity y (float)
        :param theta: Robot orientation
        :return: wheel velocity left and right (float)
        """
        velocity_x, velocity_y=velocity_y, velocity_x
        vfl = velocity_x + velocity_y + self.wheel_adjustment * omega
        vfr = velocity_x - velocity_y - self.wheel_adjustment * omega
        vrl = velocity_x - velocity_y + self.wheel_adjustment * omega
        vrr = velocity_x + velocity_y - self.wheel_adjustment * omega

        return -vfl,vfr,-vrl,vrr

    def initialize(self, robot_index, client_id):
        self.robot_index, self.client_id = robot_index, client_id
        (
            robot_handle,
            omniRob_FL_handle,omniRob_FR_handle,omniRob_RL_handle, omniRob_RR_handle
        ) = vrep_interface_holonomic.get_vrep_handle(self.client_id, self.robot_index)
        self.robot_handle = robot_handle
        self.omniRob_FL_handle,self.omniRob_FR_handle,self.omniRob_RL_handle, self.omniRob_RR_handle=omniRob_FL_handle,omniRob_FR_handle,omniRob_RL_handle, omniRob_RR_handle
    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in simulator/real world
        :param control_data: Controls to be executed
        """
        velocity_x_local = control_data.velocity_x
        velocity_y_local = control_data.velocity_y
        theta_global = control_data.orientation[2]
        omega=control_data.omega
        velocity_x_global=velocity_x_local*math.cos(theta_global)-velocity_y_local*math.sin(theta_global)
        velocity_v_global=velocity_x_local*math.sin(theta_global)+velocity_y_local*math.cos(theta_global)
        # velocity_x_global, velocity_v_global,omega=1,0,0
        vfl,vfr,vrl,vrr = self.velocity_transform(velocity_x_global, velocity_v_global, theta_global,omega)

        # omega_left = 1 * self.wheel_adjustment
        # omega_right = -1 * self.wheel_adjustment


        vrep_interface_holonomic.post_control(
            self.client_id,
            self.omniRob_FL_handle,
            self.omniRob_FR_handle,
            self.omniRob_RL_handle,
            self.omniRob_RR_handle,
            vfl,
            vfr,
            vrl,
            vrr
        )
        # vrep_interface.synchronize(self.client_id)
