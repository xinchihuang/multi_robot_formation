"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""
from vrep import vrep_interface
import math


class Executor:
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
        self.motor_left_handle = None
        self.motor_right_handle = None
        self.point_cloud_handle = None

        self.centralized_k = 1
        self.max_velocity = 0.5
        self.wheel_adjustment = 10.25

    def velocity_transform(self, velocity_x, velocity_y, theta):
        """
        Transform robot velocity to wheels velocity
        :param velocity_x:  robot velocity x (float)
        :param velocity_y: robot velocity y (float)
        :param theta: Robot orientation
        :return: wheel velocity left and right (float)
        """
        kk = self.centralized_k
        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        wheel_velocity_left = M11 * velocity_x + M12 * velocity_y
        wheel_velocity_right = M21 * velocity_x + M22 * velocity_y

        if (
            math.fabs(wheel_velocity_right) >= math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_right) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_right)
        elif (
            math.fabs(wheel_velocity_right) < math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_left) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_left)
        else:
            alpha = 1

        wheel_velocity_left = alpha * wheel_velocity_left
        wheel_velocity_right = alpha * wheel_velocity_right
        return wheel_velocity_left, wheel_velocity_right

    def initialize(self, robot_index, client_id):
        self.robot_index, self.client_id = robot_index, client_id
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
        :param control_data: Controls to be executed
        """
        velocity_x_local = control_data.velocity_x
        velocity_y_local = control_data.velocity_y
        theta_global = control_data.orientation[2]
        velocity_x_global=velocity_x_local*math.cos(theta_global)-velocity_y_local*math.sin(theta_global)
        velocity_v_global=velocity_x_local*math.sin(theta_global)+velocity_y_local*math.cos(theta_global)
        omega_left, omega_right = self.velocity_transform(velocity_x_global, velocity_v_global, theta_global)
        omega_left = omega_left * self.wheel_adjustment
        omega_right = omega_right * self.wheel_adjustment
        vrep_interface.post_control(
            self.client_id,
            self.motor_left_handle,
            self.motor_right_handle,
            omega_left,
            omega_right,
        )
        # vrep_interface.synchronize(self.client_id)
