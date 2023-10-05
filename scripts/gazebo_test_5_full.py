#!/usr/bin/env python3

import random

import numpy as np
import rospy
import time
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import cv2
import os
import message_filters
import collections
from squaternion import Quaternion

from utils.gabreil_graph import get_gabreil_graph,get_gabreil_graph_local,global_to_local
from utils.initial_pose import initialize_pose,PoseDataLoader
from utils.occupancy_map_simulator import MapSimulator
from model.LocalExpertController import LocalExpertController,LocalExpertControllerHeuristic
from controller_new import VitController

class Simulation:
    def __init__(self, robot_num,controller=None,save_data_root=None):
        self.robot_num=robot_num
        self.sub_topic_list = []
        self.pub_topic_dict = collections.defaultdict()
        # for index in range(self.robot_num):
        #     point_topic=f"D435_camera_{index}/depth/color/points"
        #     self.sub_topic_list.append(message_filters.Subscriber(point_topic, PointCloud2))
        for index in range(self.robot_num):
            pose_topic=f'rm_{index}/odom'
            self.sub_topic_list.append(message_filters.Subscriber(pose_topic, Odometry))
        for index in range(self.robot_num):
            pub_topic=f'rm_{index}/cmd_vel'
            self.pub_topic_dict[index]=rospy.Publisher(pub_topic, Twist, queue_size=10)
        # print(self.sub_topic_list)
        self.controller=controller
        ts = message_filters.ApproximateTimeSynchronizer(self.sub_topic_list, queue_size=10, slop=0.1,allow_headerless=True)
        ts.registerCallback(self.SimulateCallback)

        self.save_data_root=save_data_root
        self.robot_upper_bound=0.12
        self.robot_lower_bound=-0.12
        self.map_size = 100
        self.max_simulation_time_step=2000

        self.sensor_range=5
        self.sensor_angle=2*math.pi
        self.max_velocity=0.5

        self.desired_distance=2
        self.trace=[]
        self.observation_list=[]
        self.reference_control=[]
        self.time_step=0

    def save_to_file(self):
        root=self.save_data_root
        if not os.path.exists(root):
            os.mkdir(root)
        num_dirs = len(os.listdir(root))
        data_path = os.path.join(root, str(num_dirs))
        os.mkdir(data_path)
        # observation_array=np.array(self.observation_list)
        trace_array=np.array(self.trace)
        reference_control_array=np.array(self.reference_control)
        # np.save(os.path.join(data_path,"observation.npy"),observation_array)
        np.save(os.path.join(data_path, "trace.npy"), trace_array)
        np.save(os.path.join(data_path, "reference.npy"), reference_control_array)



    def SimulateCallback(self, *argv):
        # print(argv)
        pose_list = []
        control_list=[]
        for index in range(self.robot_num):
            q=Quaternion(argv[index].pose.pose.orientation.x,argv[index].pose.pose.orientation.y,argv[index].pose.pose.orientation.z,argv[index].pose.pose.orientation.w)
            pose_index=[argv[index].pose.pose.position.x,argv[index].pose.pose.position.y,q.to_euler(degrees=False)[0]]
            pose_list.append(pose_index)
        self.trace.append(pose_list)
        print("___________")
        # print(pose_list)
        gabreil_graph_global=get_gabreil_graph(pose_list)
        for i in range(len(gabreil_graph_global)):
            for j in range(i+1,len(gabreil_graph_global)):
                if gabreil_graph_global[i][j] == 0:
                    continue
                distance = ((pose_list[i][0] - pose_list[j][0]) ** 2 + (
                            pose_list[i][1] - pose_list[j][1]) ** 2) ** 0.5
                print(i, j, distance)
            occupancy_map_simulator = MapSimulator(max_x=self.sensor_range, max_y=self.sensor_range,sensor_view_angle= self.sensor_angle, local=True,partial=False)
            position_lists_local=global_to_local(pose_list)
            for index in range(0, self.robot_num):
                # print(position_lists_local[index])
                occupancy_map = occupancy_map_simulator.generate_map_one(position_lists_local[index])
                # cv2.imshow("robot view " + str(index), np.array(occupancy_map))
                # cv2.waitKey(1)
                control_data = self.controller.get_control(index, occupancy_map)
                control_list.append([control_data.velocity_x, control_data.velocity_y, control_data.omega])
            # print(control_list)
        for index in range(0,self.robot_num):
            msg=Twist()
            msg.linear.x = control_list[index][0] if abs(control_list[index][0])<self.max_velocity else self.max_velocity*abs(control_list[index][0])/control_list[index][0]
            msg.linear.y = control_list[index][1] if abs(control_list[index][1])<self.max_velocity else self.max_velocity*abs(control_list[index][1])/control_list[index][1]
            self.pub_topic_dict[index].publish(msg)
        self.time_step+=1

        if self.time_step>self.max_simulation_time_step:
            print("save")
            self.save_to_file()
            rospy.signal_shutdown(f"Stop after {self.time_step} steps")





if __name__ == "__main__":
    pose_list=initialize_pose(5)
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    rospy.init_node("collect_data")
    robot_num = 5
    ### Vit controller
    model_path="/home/xinchi/vit_full/vit.pth"
    save_data_root="/home/xinchi/gazebo_data/ViT_5_full"
    controller=VitController(model_path)


 #
 #    pose_list=[[-1.8344854  ,-2.54902913  ,1.31531797],
 # [-0.11962687 ,-2.94522615 ,-2.78613711],
 # [-4.51360495  ,1.04370626  ,0.72373201],
 # [ 0.34727331  ,1.90429804 ,-1.54858546],
 # [-2.34736724  ,2.89713682 ,-1.14321162]]

    listener = Simulation(robot_num,controller)

    for i in range(len(pose_list)):
        state_msg = ModelState()
        state_msg.model_name = 'rm_{}'.format(i)
        state_msg.pose.position.x = pose_list[i][0]
        state_msg.pose.position.y = pose_list[i][1]
        state_msg.pose.position.z = 0
        q=Quaternion.from_euler(0, 0, pose_list[i][2], degrees=False)
        state_msg.pose.orientation.x = q.x
        state_msg.pose.orientation.y = q.y
        state_msg.pose.orientation.z = q.z
        state_msg.pose.orientation.w = q.w
        resp = set_state(state_msg)
    rospy.spin()

