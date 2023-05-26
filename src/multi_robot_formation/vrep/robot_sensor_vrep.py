"""
A sensor template. Get information from simulator/real-world
author: Xinchi Huang
"""
import collections
import math
import numpy as np

from vrep import vrep_interface
from comm_data import SensorData

# from ..utils.occupancy_map_simulator import MapSimulator

class Sensor:
    """
    Robot sensor
    """

    def __init__(self):
        self.client_id = None
        self.robot_handle = None
        self.robot_index = None

        #### sensor range related settings
        self.max_x = 10
        self.max_y = 10
        self.max_height = 0.3
        self.min_range = 0.2

        #### sensor output settings
        self.occupancy_map_size = 100
        self.point_cloud = []
        self.sensor_buffer = []
        self.sensor_buffer_count = 0

    def filter_data(self, world_point):
        x = world_point[0]
        y = world_point[1]
        z = world_point[2]
        if (
            x > self.max_x
            or x < -self.max_x
            or y > self.max_y
            or y < -self.max_y
            or z < -self.max_height
        ):  #
            return None
        elif (
            x < self.min_range
            and y < self.min_range
            and x > -self.min_range
            and y > -self.min_range
        ):
            return None
        return world_point

    def world_to_map(self, world_point):
        if world_point == None:
            return None
        x_world = world_point[0]
        y_world = world_point[1]

        y_map = min(int(self.occupancy_map_size / 2) + int(x_world * self.occupancy_map_size / self.max_x / 2), self.occupancy_map_size - 1)
        x_map = min(int(self.occupancy_map_size / 2) - int(y_world * self.occupancy_map_size / self.max_y / 2), self.occupancy_map_size - 1)
        return [x_map, y_map]

    def process_raw_data(self, point_cloud):
        sensor_points = point_cloud
        occupancy_map = (
            np.ones((self.occupancy_map_size, self.occupancy_map_size)) * 255
        )
        # print(occupancy_map)
        # print(len(sensor_points))
        for i in range(0, len(sensor_points), 3):
            x_world = sensor_points[i + 0]
            y_world = sensor_points[i + 2]
            z_world = sensor_points[i + 1]

            x_world,y_world=y_world,x_world
            point_world = self.filter_data([x_world, y_world, z_world])
            point_map = self.world_to_map(point_world)
            if point_map == None:
                continue
            # print("world point",self.robot_index)
            # print(x_world,y_world)
            # print("map point of robot", self.robot_index, self.robot_handle)
            # print(point_map)
            occupancy_map[point_map[0]][point_map[1]] = 0
        return occupancy_map

    def process_raw_data_new(self, point_cloud):
        sensor_points = point_cloud
        occupancy_map = (
                np.ones((self.occupancy_map_size, self.occupancy_map_size)) * 255
        )
        # print(occupancy_map)
        group=[]
        for i in range(0, len(sensor_points), 3):
            x_world = sensor_points[i + 0]
            y_world = sensor_points[i + 2]
            z_world = sensor_points[i + 1]

            x_world, y_world = y_world, x_world
            point_world = self.filter_data([x_world, y_world, z_world])
            self.preprocess_point(point_world,group)
        # print(group)
        for i in range(len(group)):
            point_center=group[i][0]
            point_map = self.world_to_map(point_center)
            # print("world point",self.robot_index)
            # print(x_world,y_world)
            # print("map point of robot", self.robot_index, self.robot_handle)
            # print(point_map)
            occupancy_map[point_map[0]][point_map[1]] = 0
            occupancy_map[point_map[0]+1][point_map[1]] = 0
            occupancy_map[point_map[0]][point_map[1]+1] = 0
            occupancy_map[point_map[0]+1][point_map[1]+1] = 0
        return occupancy_map
    def preprocess_point(self,point,group):
        if point==None:
            return
        ###(center,number)
        x_world=point[0]
        y_world=point[1]
        for i in range(len(group)):
            center=group[i][0]
            number=group[i][1]
            distance=((x_world-center[0])**2+(y_world-center[1])**2)**0.5

            if distance<(2*self.min_range):

                new_center_x = (number * center[0] + x_world) / (number + 1)
                new_center_y = (number * center[1] + y_world) / (number + 1)

                del group[i]

                new_item=((new_center_x,new_center_y),number+1)
                group.append(new_item)
                return
        new_item = ((x_world, y_world), 1)
        group.append(new_item)
        return

    def get_sensor_data(self):
        """
        Get sensor readings
        :return: Data from sensor and simulator
        """
        robot_sensor_data = SensorData()
        position, orientation = vrep_interface.get_robot_pose(
            self.client_id, self.robot_handle
        )
        (
            linear_velocity,
            angular_velocity,
            velodyne_points,
        ) = vrep_interface.get_sensor_data(
            self.client_id, self.robot_handle, self.robot_index
        )

        self.sensor_buffer.extend(velodyne_points[2])
        self.sensor_buffer_count += 1
        # print(self.sensor_buffer_count)
        if self.sensor_buffer_count == 4:
            self.point_cloud = self.sensor_buffer.copy()
            self.sensor_buffer.clear()
            self.sensor_buffer_count = 0

        # print("points ",len(self.point_cloud))
        robot_sensor_data.robot_index = self.robot_index
        robot_sensor_data.position = position
        robot_sensor_data.orientation = orientation
        robot_sensor_data.linear_velocity = linear_velocity
        robot_sensor_data.angular_velocity = angular_velocity

        occupancy_map = self.process_raw_data(self.point_cloud)

        # occupancy_map = self.process_raw_data_new(self.point_cloud)

        # ### fake data
        # global_positions = [[-epoch5, -epoch5, 0], [-epoch5, epoch5, 0], [epoch5, epoch5, 0], [epoch5, -epoch5, 0], [0, 0, 0]]
        # position_lists_local = occupancy_map_simulator.global_to_local(global_positions)
        # robot_size, max_height, map_size, max_x, max_y = 0.2, 0.epoch5, 100, 4, 4
        # occupancy_map = occupancy_map_simulator.generate_map(
        #     position_lists_local, robot_size, max_height, map_size, max_x, max_y
        # )
        robot_sensor_data.occupancy_map = occupancy_map

        return robot_sensor_data
