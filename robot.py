
"""
A robot template
author: Xinchi Huang
"""
from robot_info_vrep import info
class robot():
    def __init__(self):
        self.position=None
        self.orientation=None
        self.max_velocity = 1.2
        self.sensor_data=None
        self.info=info()
    def update_pose(self,position,orientation):
        self.position=position
        self.orientation=orientation
    def get_sensor_data(self,data):
        self.sensor_data=data
    def get_control(self):
        return 0,0