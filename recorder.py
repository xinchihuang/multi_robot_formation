"""
A recorder template. Used for recording simulation data
author: Xinchi Huang
"""
class recorder():
    def __init__(self):
        self.data=None
    def record_sensor_data(self):
        return True
    def record_robot_pose(self):
        return True
    def record_controller_output(self):
        return True