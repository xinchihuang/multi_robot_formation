"""
A scene template
author: Xinchi Huang
"""
import vrep_interface
from robot import robot
class scene():
    def __init__(self,num_robot):
        self.num_robots=num_robot
        self.robot_list=[]
        self.adjacency_matrix=[]
        self.client_id=None
    def initial_vrep(self):
        self.client_id=vrep_interface.init_vrep()
    def add_robot(self,robot_index):
        new_robot=robot()
        robot_handle, motor_left_handle,\
        motor_right_handle, point_cloud_handle=vrep_interface.get_vrep_handle(self.client_id,robot_index)

        new_robot.info.client_id = self.client_id
        new_robot.info.robot_handle=robot_handle
        new_robot.info.motor_left_handle=motor_left_handle
        new_robot.info.motor_right_handle=motor_right_handle
        new_robot.info.point_cloud_handle=point_cloud_handle

        self.robot_list.append(new_robot)
        
    def update_adjacency_matrix(self):
        return None
    def reset_positions(self,disp_range):
        return None
