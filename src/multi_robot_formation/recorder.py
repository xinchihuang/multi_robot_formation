"""
A recorder template. Used for recording simulation data
author: Xinchi Huang
"""
import os
import numpy as np
from collections import defaultdict
from plots.plot_scene import plot_load_data


class Recorder:
    def __init__(self):
        self.root_dir = None
        self.sensor_data = defaultdict(list)
        self.trace_data = defaultdict(list)
        self.control_data = defaultdict(list)

    def record_sensor_data(self, sensor_data):
        """

        :param sensor_data: Sensor data from robot sensor and simulator. Defined in robot_sensor_realsense.py(SensorData)
        :return:
        """
        self.sensor_data[sensor_data.robot_index].append(sensor_data.occupancy_map)

    def record_robot_trace(self, sensor_data):
        """
        :param sensor_data: Sensor data from robot sensor and simulator. Defined in robot_sensor_realsense.py(SensorData)
        :return:
        trace_data:[ [position],
                     [orientation],
                     [linear_velocity],
                     [angular_velocity] ]
        """
        trace_data = []
        trace_data.append(sensor_data.position)
        trace_data.append(sensor_data.orientation)
        trace_data.append(sensor_data.linear_velocity)
        trace_data.append(sensor_data.angular_velocity)
        self.trace_data[sensor_data.robot_index].append(trace_data)

    def record_controller_output(self, control_data):
        """

        :param control_data: Control data from controller. Defined in controller.py (ControlData)
        :return:
        """

        self.control_data[control_data.robot_index].append(
            [control_data.velocity_x, control_data.velocity_y]
        )

    def save_to_file(self):
        """
        Save recorded data to files
        :return:
        """

        present = os.getcwd()
        root = os.path.join(present, self.root_dir)

        if not os.path.exists(root):
            os.mkdir(root)
        num_dirs = len(os.listdir(root))
        simulation_path = os.path.join(root, str(num_dirs))
        os.mkdir(simulation_path)

        for robot_index in self.sensor_data:
            robot_path = os.path.join(simulation_path, str(robot_index))
            os.mkdir(robot_path)
            # save lidar readings
            sensor_data_path = os.path.join(robot_path, "points.npy")
            sensor_data = self.sensor_data[robot_index]
            sensor_data_array = np.array(sensor_data)
            np.save(sensor_data_path, sensor_data_array)
            # save trace
            trace_data_path = os.path.join(robot_path, "trace.npy")
            trace_data = self.trace_data[robot_index]
            trace_data_array = np.array(trace_data)
            np.save(trace_data_path, trace_data_array)
            # save control
            control_data_path = os.path.join(robot_path, "control.npy")
            control_data = self.control_data[robot_index]
            control_data_array = np.array(control_data)
            np.save(control_data_path, control_data_array)
        plot_load_data(simulation_path)
        return True
