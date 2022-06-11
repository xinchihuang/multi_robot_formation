"""
A recorder template. Used for recording simulation data
author: Xinchi Huang
"""
import os
from collections import defaultdict

class Recorder:
    def __init__(self):
        self.root_dir=None
        self.sensor_data=defaultdict(list)
        self.trace_data=defaultdict(list)
        self.control_data=defaultdict(list)
    def record_sensor_data(self,sensor_data):
        """

        :param sensor_data: Sensor data from robot sensor and simulator. Defined in robot_sensor_vrep.py(SensorData)
        :return:
        """
        self.sensor_data[sensor_data.robot_index].append(sensor_data.velodyne_points)

    def record_robot_trace(self,sensor_data):
        """
        :param sensor_data: Sensor data from robot sensor and simulator. Defined in robot_sensor_vrep.py(SensorData)
        :return:
        """
        trace_data=[]
        trace_data.append(sensor_data.position)
        trace_data.append(sensor_data.orientation)
        trace_data.append(sensor_data.linear_velocity)
        trace_data.append(sensor_data.angular_velocity)
        self.trace_data[sensor_data.robot_index].append(trace_data)

    def record_controller_output(self,control_data):
        """

        :param control_data: Control data from controller. Defined in controller.py (ControlData)
        :return:
        """

        self.control_data[control_data.robot_index].append([control_data.omega_left,control_data.omega_right])

    def save_to_file(self):
        present=os.getcwd()
        root=os.path.join(present,self.root_dir)
        print(root)
        if os.path.exists(root):
            print(root)



        return True
