#!/usr/bin/env python3

"""
A sensor template. Get information from simulator/real-world
author: Xinchi Huang
"""
import math
import numpy
import numpy as np


import occupancy_map_simulator
import pyrealsense2 as rs
import cv2
from comm_data import SensorData


def get_frame():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image)
            )  # depth image is 1 channel, color is 3 channels
            bg_removed = np.where(
                (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                grey_color,
                color_image,
            )

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


# class SensorData:
#     """
#     A class for record sensor data
#     """
#
#     def __init__(self):
#         self.robot_index = None
#         self.position = None
#         self.orientation = None
#         self.linear_velocity = None
#         self.angular_velocity = None
#         self.occupancy_map = None


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
        x_map = int((self.max_x - x_world) / (2 * self.max_x) * self.occupancy_map_size)
        y_map = int((self.max_y - y_world) / (2 * self.max_y) * self.occupancy_map_size)

        return [x_map, y_map]

    def process_raw_data(self, point_cloud):
        sensor_points = point_cloud
        occupancy_map = (
            np.ones((self.occupancy_map_size, self.occupancy_map_size)) * 255
        )
        # print(occupancy_map)

        for i in range(0, len(sensor_points), 3):
            x_world = sensor_points[i + 0]
            y_world = sensor_points[i + 2]
            z_world = sensor_points[i + 1]
            # print("world point of robot", self.robot_index)
            # print([x_world,y_world,z_world])
            # if self.robot_index==2:
            #     print([x_world,y_world,z_world])
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

    def get_sensor_data(self):
        """
        Get sensor readings
        :return: Data from sensor and simulator
        """
        robot_sensor_data = SensorData()

        linear_velocity = 0
        angular_velocity = 0
        position = None
        orientation = None
        # occupancy_map=self.process_raw_data(point_cloud)
        ### fake data

        global_positions = [[-4, -4, 0], [-4, 4, 0], [4, 4, 0], [4, -4, 0], [0, 0, 0]]
        position_lists_local = occupancy_map_simulator.global_to_local(global_positions)
        robot_size, max_height, map_size, max_x, max_y = 0.2, 0.3, 100, 10, 10
        occupancy_map = occupancy_map_simulator.generate_map(
            position_lists_local, robot_size, max_height, map_size, max_x, max_y
        )
        position = global_positions[self.robot_index]
        orientation = global_positions[self.robot_index]
        robot_sensor_data.robot_index = self.robot_index
        robot_sensor_data.position = position
        robot_sensor_data.orientation = orientation
        robot_sensor_data.linear_velocity = linear_velocity
        robot_sensor_data.angular_velocity = angular_velocity
        robot_sensor_data.occupancy_map = occupancy_map[self.robot_index]

        return robot_sensor_data
