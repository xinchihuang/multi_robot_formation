#!/usr/bin/env python3
import numpy as np
import rospy
import sys
import os
from multi_robot_formation.realrobot.robot_executor_robomaster import Executor
from multi_robot_formation.comm_data import SceneData, SensorData,ControlData
from multi_robot_formation.controller import Controller
from multi_robot_formation.robot_template import Robot

# # from robot_test import *
from collections import defaultdict
import time

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from cmvision.msg import Blobs
from cmvision_3d.msg import Blobs3d, Blob3d

print(os.getcwd())

class ModelControl:
    def __init__(self, topic):
        self.robot = Robot(
            sensor=None,
            controller=Controller(),
            executor=Executor(),
            platform="robomaster",
            controller_type="model",
        )
        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.ModelControlCallback)
        self.map_size = 100
        self.range = 5
        self.height = 2
        self.color_index = {"red": 0, "yellow": 1, "green": 2}
        self.model_path = os.path.join(
            os.getcwd()
            + "/src/multi_robot_formation/src/multi_robot_formation/saved_model/model_dummy.pth"
        )
        self.EP_DICT = {}
        self.IP_DICT = {0: "172.20.10.6", 1: "172.20.10.7", 2: "172.20.10.8"}
        # self.robot.controller.initialize_GNN_model(1, self.model_path)
        # self.IP_DICT={1:'172.20.10.7'}

        # for index,ip in self.IP_DICT.items():
        #     print('%s connecting...' % ip)
        #     self.EP_DICT[ip] = EP(ip)
        #     self.EP_DICT[ip].start()

    def initialize_robot(self):
        controller = Controller()
        executor = Executor()
        self.robot = Robot(
            sensor=None,
            controller=None,
            executor=None,
            platform="robomaster",
            controller_type="model",
        )
        self.robot.executor = executor
        self.robot.controller = controller
        self.robot.controller.initialize_GNN_model(1, self.model_path)
    def simple_control(self,position_list,index,desired_distance):
        out_put = ControlData()
        velocity_sum_x=0
        velocity_sum_y=0
        for i in range(len(position_list)):
            x=position_list[i][0]
            y=position_list[i][1]
            distance=(x**2+y**2)**0.5
            rate = (distance - desired_distance) / distance
            velocity_x = rate * x
            velocity_y = rate * y
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        out_put.robot_index = index
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y
        return out_put
    def ModelControlCallback(self, data):
        print("ros_initialized")
        position_list_local = []
        look_up_table = [0, 0, 0]
        for blob in data.blobs:
            if not blob.name in self.color_index:
                continue
            robot_index = self.color_index[blob.name]
            if look_up_table[robot_index] == 1:
                continue
            look_up_table[robot_index] = 1
            x_c, y_c, z_c = blob.center.x, blob.center.y, blob.center.z
            # print(blob.name,x_w,y_w,z_w)
            position_list_local.append([x_c, y_c, z_c])
        if len(position_list_local) == 0:
            print("no data")
        print(position_list_local)
        model_data=self.simple_control(position_list_local,0,1)

        # model_data = self.robot.controller.decentralized_control_real(
        #     index=0, position_lists_local=position_list_local
        # )
        # model_data.velocity_x=1
        # model_data.velocity_y=1
        self.robot.executor.execute_control(model_data)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         if i==j:
        #             continue
        #         distance = ((position_dict[i][0] - position_dict[j][0]) ** 2
        #                            + (position_dict[i][1] - position_dict[j][1]) ** 2
        #                    ) ** 0.5
        #         adjacency_list[i].append((j,position_dict[j][0],position_dict[j][1],distance))
        # scene_data.adjacency_list=adjacency_list
        # # print("AAAAAAAAAAAAA")
        #
        # for index, ip in self.IP_DICT.items():
        #     print(ip)
        #     control_data=centralized_control(index, sensor_data_list[index], scene_data)
        #     print(control_data.omega_left,control_data.omega_right)
        #     self.EP_DICT[ip].command('chassis speed x '+ str(control_data.omega_right)+' y '+str(control_data.omega_left)+' z 0')
        #     # self.EP_DICT[ip].command('chassis speed x 0 y 0 z 0')
        # # self.executor.execute_control(control_data)


if __name__ == "__main__":
    rospy.init_node("model_control")
    topic = "/blobs_3d"
    listener = ModelControl(topic)
    time.sleep(1)
    rospy.spin()
