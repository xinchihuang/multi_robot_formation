#!/usr/bin/env python3
import math
import rospy
import os
import cv2
from scripts.realrobots.robot_executor_robomaster import Executor
from .multi_robot_formation.comm_data import SceneData, SensorData,ControlData
from .multi_robot_formation.controller_new import VitController
from .multi_robot_formation.utils.occupancy_map_simulator import MapSimulator
from .multi_robot_formation.model.LocalExpertController import LocalExpertController
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

        self.model_path =os.path.abspath('..')+"/home/xinchi/catkin_ws/src/multi_robot_formation/scripts/saved_model/model_3200_epoch10.pth"
        self.desired_distance=1.0
        self.controller=VitController(model_path=self.model_path,desired_distance=self.desired_distance)
        # self.controller=LocalExpertController()

        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.ModelControlCallback)
        self.map_size = 100
        self.sensor_range = 2
        self.map_simulator = MapSimulator(max_x=self.sensor_range, max_y=self.sensor_range,
                                          sensor_view_angle=math.pi * 2, local=True, partial=False)
        self.executor=Executor()
        self.color_index = {"Yellow": 0,"Blue": 1,"Orange": 2,"Green": 3}
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
            x_c, z_c, y_c = blob.center.x, -blob.center.y, blob.center.z
            if -0.25<z_c<0.25:
            # print(blob.name,x_w,y_w,z_w)
                position_list_local.append([y_c, -x_c, z_c])
        if len(position_list_local) == 0:
            print("no data")
            control_data=ControlData()
        else:

            occupancy_map = self.map_simulator.generate_map_one(position_list_local)
            cv2.imshow("robot view " + str(index), np.array(occupancy_map))
            cv2.waitKey(1)
            data = {"robot_id": 0, "occupancy_map": occupancy_map}
            control_data = self.controller.get_control(data)

        self.executor.execute_control(control_data=control_data)

    def keyboard_stop(self):
        if data.data == 'q':
            self.robot.executor.stop()
            # exit(1)
            rospy.signal_shutdown("Shut down!")
def stop_node(event):
    rospy.signal_shutdown("Time's up!")
if __name__ == "__main__":
    # signal.signal(signal.SIGINT, handler)
    rospy.init_node("model_control")
    topic = "/blobs_3d"
    listener = ModelControl(topic)
    rospy.Subscriber('keyboard_input', String, listener.keyboard_stop)
    # timer = rospy.Timer(rospy.Duration(100), listener.timed_stop)
    rospy.spin()

