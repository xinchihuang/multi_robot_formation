#!/usr/bin/env python3
import numpy as np
import rospy
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import cv2


class DataCollector:
    def __init__(self, listen_topic,pub_topic):
        self.topic = listen_topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.topic, PointCloud2, self.DataCollectorCallback)
        self.map_size = 100
        self.range = 5
        self.height = 2
        self.pub= rospy.Publisher(pub_topic, Twist, queue_size=10)

    def point_to_map(self, points):

        occupancy_map = np.ones((self.map_size, self.map_size))
        for point in points:
            x_map = int((-point[2] / self.range) * self.map_size/2 + self.map_size / 2)
            y_map = int((point[0] / self.range) * self.map_size/2 + self.map_size / 2)
            if 0 <= x_map < self.map_size and 0 <= y_map < self.map_size:
                occupancy_map[x_map][y_map] = 0
        return occupancy_map

    def DataCollectorCallback(self, data):
        # print(data)
        points = []
        for point in point_cloud2.read_points(data, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            if 0.1<pt_y < 0.18:
                # print([pt_x, pt_y, pt_z])
                points.append([pt_x, pt_y, pt_z])
        occupancy_map = self.point_to_map(points)
        cv2.imshow("Example occupancy_map", occupancy_map)
        key = cv2.waitKey(1)
        msg=Twist()
        msg.linear.x = 1
        msg.linear.y = 0
        msg.linear.x = 0
        msg.angular.z = 1
        speed = 0
        self.pub.publish(msg)



if __name__ == "__main__":

    rospy.init_node("collect_data0")
    listen_topic = "/D435_camera_0/depth/color/points"
    pub_topic='rm_0/cmd_vel'
    listener = DataCollector(listen_topic,pub_topic)
    rospy.spin()

    rospy.init_node("collect_data4")
    listen_topic = "/D435_camera_4/depth/color/points"
    pub_topic = 'rm_4/cmd_vel'
    listener = DataCollector(listen_topic, pub_topic)
    rospy.spin()
