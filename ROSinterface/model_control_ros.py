#!/usr/bin/env python3
import numpy as np
import rospy
import sys
import os
import signal
from multi_robot_formation.realrobot.robot_executor_robomaster import Executor
from multi_robot_formation.comm_data import SceneData, SensorData,ControlData
from multi_robot_formation.controller_new import VitController
from multi_robot_formation.robot_template import Robot
from multi_robot_formation.utils.occupancy_map_simulator import MapSimulator
from multi_robot_formation.model.LocalExpertController import LocalExpertController
# # from robot_test import *
from collections import defaultdict
import time

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from cmvision.msg import Blobs
from cmvision_3d.msg import Blobs3d, Blob3d


class ModelControl:
    def __init__(self, topic):

        self.model_path =os.path.abspath('..')+"/jetson/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit1.0.pth"
        self.desired_distance=1.0

        # self.controller=VitController(model_path=self.model_path,desired_distance=self.desired_distance)

        self.controller=LocalExpertController()
        self.robot = Robot(
            sensor=None,
            executor=Executor(),
            controller=self.controller,
            platform="robomaster",
        )

        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.ModelControlCallback)
        self.map_size = 100
        self.range = 5
        self.height = 2
        self.color_index = {"green": 0}
        self.EP_DICT = {}
        # self.IP_DICT = {0: "172.20.4.epoch1_3000", 1: "172.20.4.7", 2: "172.20.4.8"}
        # self.robot.controller.initialize_GNN_model(1, self.model_path)
        # self.IP_DICT={1:'172.20.4.7'}

        # for index,ip in self.IP_DICT.items():
        #     print('%s connecting...' % ip)
        #     self.EP_DICT[ip] = EP(ip)
        #     self.EP_DICT[ip].start()
    def simple_control(self,position_list,index,desired_distance):
        out_put = ControlData()
        velocity_sum_x=0
        velocity_sum_y=0
        for i in range(len(position_list)):
            x=position_list[i][0]
            y=position_list[i][1]
            distance=(x**2+y**2)**0.5
            rate = (distance - desired_distance) / distance

            velocity_x = rate * (-x)
            velocity_y = rate * (-y)
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        out_put.robot_index = index
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y
        return out_put
    def ModelControlCallback(self, data):
        position_list_local = []
        look_up_table = [0, 0, 0]
        for blob in data.blobs:
            if not blob.name in self.color_index:
                continue
            robot_index = self.color_index[blob.name]
            # if look_up_table[robot_index] == 1:
            #     continue
            look_up_table[robot_index] = 1
            x_c, z_c, y_c = blob.center.x, -blob.center.y, -blob.center.z
            if -0.2<z_c<0.2:
            # print(blob.name,x_w,y_w,z_w)
                position_list_local.append([x_c, y_c, z_c])
        if len(position_list_local) == 0:
            print("no data")
            control_data=ControlData()
        else:
            # print("position", position_list_local)
            print(self.robot.controller.name)
            # occupancy_map_simulator = MapSimulator()
            # occupancy_map = occupancy_map_simulator.generate_map_one(position_list_local)
            # model_data=self.simple_control(position_list_local,0,1)
            # self.robot.controller.num_robot=epoch5
            # model_data=self.robot.controller.get_control(0,occupancy_map)
            position_list_local.append([0,0,0])
            print("position", position_list_local)
            control_data=self.robot.controller.get_control(len(position_list_local)-1,position_list_local)
            print(control_data.velocity_x)
        self.robot.executor.execute_control(control_data=control_data)

    def keyboard_stop(self):
        if data.data == 'q':
            self.robot.executor.stop()
            # exit(1)
            rospy.signal_shutdown("Shut down!")
    def timed_stop(self,event):
        print("Time's up!")
        self.robot.executor.stop()
        # exit(1)
        rospy.signal_shutdown("Time's up!")
def stop_node(event):
    rospy.signal_shutdown("Time's up!")
if __name__ == "__main__":
    # signal.signal(signal.SIGINT, handler)
    rospy.init_node("model_control")
    topic = "/blobs_3d"
    listener = ModelControl(topic)
    rospy.Subscriber('keyboard_input', String, listener.keyboard_stop)
    timer = rospy.Timer(rospy.Duration(100), listener.timed_stop)
    rospy.spin()

