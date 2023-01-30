#!/usr/bin/env python3
import numpy as np
import rospy
import time
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import cv2


class DataCollector:
    def __init__(self, topic):
        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.topic, PointCloud2, self.DataCollectorCallback)
        self.map_size = 1000
        self.range = 5
        self.height = 2

    def point_to_map(self, points):
        occupancy_map = np.ones((self.map_size, self.map_size))
        for point in points:
            x_map = int((-point[2] / self.range) * self.map_size + self.map_size / 2)
            y_map = int((point[0] / self.range) * self.map_size + self.map_size / 2)
            if 0 <= x_map < self.map_size and 0 <= y_map < self.map_size:
                occupancy_map[x_map][y_map] = 0
        return occupancy_map

    def DataCollectorCallback(self, data):
        points = []
        for point in point_cloud2.read_points(data, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            if pt_y < 0.182:
                print([pt_x, pt_y, pt_z])
                points.append([pt_x, pt_y, pt_z])
        occupancy_map = self.point_to_map(points)
        cv2.imshow("Example occupancy_map", occupancy_map)
        key = cv2.waitKey(1)


if __name__ == "__main__":

    rospy.init_node("collect_data")
    topic = "/camera/a/depth/color/points"
    listener = DataCollector(topic)
    rospy.spin()
