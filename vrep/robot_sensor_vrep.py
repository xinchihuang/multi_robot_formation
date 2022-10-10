"""
A sensor template. Get information from simulator/real-world
author: Xinchi Huang
"""
import math
import numpy
import numpy as np

from vrep import vrep_interface
import occupancy_map_simulator

class SensorData:
    """
    A class for record sensor data
    """

    def __init__(self):
        self.robot_index = None
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.occupancy_map = None


class Sensor:
    """
    Robot sensor
    """

    def __init__(self):
        self.client_id = None
        self.robot_handle = None
        self.robot_index = None

        #### sensor range related settings
        self.max_x=10
        self.max_y=10
        self.max_height=0.3
        self.min_range=0.2

        #### sensor output settings
        self.occupancy_map_size=100

    def filter_data(self,world_point):
        x=world_point[0]
        y=world_point[1]
        z=world_point[2]
        if x > self.max_x or x < -self.max_x or y > self.max_y or y < -self.max_y or z < -self.max_height:  #
            return None
        elif x < self.min_range and y < self.min_range and x > -self.min_range and y > -self.min_range:
            return None
        return world_point
    def world_to_map(self,world_point):
        if world_point==None:
            return None
        x_world = world_point[0]
        y_world = world_point[1]
        x_map=int((self.max_x-x_world)/(2*self.max_x)*self.occupancy_map_size)
        y_map=int((self.max_y-y_world)/(2*self.max_y)*self.occupancy_map_size)

        return [x_map,y_map]
    def process_raw_data(self,point_cloud):
        sensor_points=point_cloud
        occupancy_map=np.ones((self.occupancy_map_size,self.occupancy_map_size))*255
        # print(occupancy_map)

        for i in range(0,len(sensor_points),3):
            x_world=sensor_points[i+0]
            y_world=sensor_points[i+2]
            z_world=sensor_points[i+1]
            # print("world point of robot", self.robot_index)
            # print([x_world,y_world,z_world])
            # if self.robot_index==2:
            #     print([x_world,y_world,z_world])
            point_world=self.filter_data([x_world,y_world,z_world])
            point_map=self.world_to_map(point_world)
            if point_map==None:
                continue
            # print("world point",self.robot_index)
            # print(x_world,y_world)
            # print("map point of robot", self.robot_index, self.robot_handle)
            # print(point_map)
            occupancy_map[point_map[0]][point_map[1]]=0
        return occupancy_map

    def get_sensor_data(self):
        """
        Get sensor readings
        :return: Data from sensor and simulator
        """
        robot_sensor_data = SensorData()
        position, orientation = vrep_interface.get_robot_pose(
            self.client_id, self.robot_handle
        )
        point_cloud=[]
        linear_velocity=0
        angular_velocity=0
        for i in range(1):
            (
                linear_velocity,
                angular_velocity,
                velodyne_points,
            ) = vrep_interface.get_sensor_data(
                self.client_id, self.robot_handle, self.robot_index
            )
            point_cloud.extend(velodyne_points[2])
            # vrep_interface.synchronize(self.client_id)
        # print(len(point_cloud))
        robot_sensor_data.robot_index = self.robot_index
        robot_sensor_data.position = position
        robot_sensor_data.orientation = orientation
        robot_sensor_data.linear_velocity = linear_velocity
        robot_sensor_data.angular_velocity = angular_velocity

        # occupancy_map=self.process_raw_data(point_cloud)

        ### fake data
        global_positions = [[-4, -4, 0],
                            [-4, 4, 0],
                            [4, 4, 0],
                            [4, -4, 0],
                            [0, 0, 0]]
        position_lists_local = occupancy_map_simulator.global_to_local(global_positions)
        robot_size,max_height, map_size, max_x, max_y = 0.2, 0.3, 100, 10, 10
        occupancy_map=occupancy_map_simulator.generate_map(position_lists_local,robot_size,max_height,map_size,max_x,max_y)
        robot_sensor_data.occupancy_map=occupancy_map[self.robot_index]

        return robot_sensor_data
