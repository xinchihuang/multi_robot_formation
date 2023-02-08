#!/usr/bin/env python3
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

from multi_robot_formation.controller import Controller
from multi_robot_formation.comm_data import SceneData,SensorData,ControlData
from multi_robot_formation.utils.data_generator import DataGenerator
class DataCollector:
    def __init__(self, robot_num):
        self.robot_num=robot_num
        self.sub_topic_list = []
        self.pub_topic_dict = collections.defaultdict()
        for index in range(self.robot_num):
            point_topic=f"D435_camera_{index}/depth/color/points"
            self.sub_topic_list.append(message_filters.Subscriber(point_topic, PointCloud2))
        for index in range(self.robot_num):
            pose_topic=f'rm_{index}/odom'
            self.sub_topic_list.append(message_filters.Subscriber(pose_topic, Odometry))
        for index in range(self.robot_num):
            pub_topic=f'rm_{index}/cmd_vel'
            self.pub_topic_dict[index]=rospy.Publisher(pub_topic, Twist, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer(self.sub_topic_list, 100,1,allow_headerless=True)
        ts.registerCallback(self.DataCollectorCallback)

        self.save_data_root="/home/xinchi/gazebo_data"
        self.upper_bound=0.12
        self.lower_bound=-0.12
        self.map_size = 100
        self.range = 5
        self.height = 2
        self.max_time_step=1000

        self.trace=[]
        self.observation_list=[]
        self.reference_control=[]
        self.time_step=0



    def point_to_map(self, points):

        occupancy_map = np.ones((self.map_size, self.map_size))
        for point in points:
            x_map = int((-point[2] / self.range) * self.map_size/2 + self.map_size / 2)
            y_map = int((point[0] / self.range) * self.map_size/2 + self.map_size / 2)
            if 0 <= x_map < self.map_size and 0 <= y_map < self.map_size:
                occupancy_map[x_map][y_map] = 0
        return occupancy_map
    def save_to_file(self):
        root=self.save_data_root
        if not os.path.exists(root):
            os.mkdir(root)
        num_dirs = len(os.listdir(root))
        data_path = os.path.join(root, str(num_dirs))
        os.mkdir(data_path)
        observation_array=np.array(self.observation_list)
        trace_array=np.array(self.trace)
        reference_control_array=np.array(self.reference_control)
        np.save(os.path.join(data_path,"observation.npy"),observation_array)
        np.save(os.path.join(data_path, "trace.npy"), trace_array)
        np.save(os.path.join(data_path, "reference.npy"), reference_control_array)
    def expert_control_gazebo(self,pose_list,robot_id,local):
        data_generator=DataGenerator()
        controller=Controller()
        sensor_data=SensorData()
        scene_data=SceneData()
        controller.desired_distance=1.0
        adjacency_list=data_generator.update_adjacency_list(pose_list)
        sensor_data.position=pose_list[robot_id]
        sensor_data.orientation=[0,0,pose_list[robot_id][2]]
        scene_data.adjacency_list=adjacency_list
        control_data=controller.centralized_control(robot_id,sensor_data,scene_data)
        velocity_x,velocity_y=control_data.velocity_x, control_data.velocity_y
        if local:
            theta = sensor_data.orientation[2]
            velocity_x_global = velocity_x * math.cos(theta) + velocity_y * math.sin(
                theta
            )
            velocity_y_global = -velocity_x * math.sin(theta) + velocity_y * math.cos(
                theta
            )
            velocity_x = velocity_x_global
            velocity_y = velocity_y_global
        return velocity_x,velocity_y



    def DataCollectorCallback(self, *argv):
        occupancy_map_list = []
        pose_list = []
        control_list=[]
        for index in range(0,self.robot_num):
            point_data=argv[index]
            points = []
            for point in point_cloud2.read_points(point_data, skip_nans=True):
                pt_x = point[0]
                pt_y = point[1]
                pt_z = point[2]

                if self.lower_bound< pt_y < self.upper_bound:
                    # print([pt_x, pt_y, pt_z])
                    points.append([pt_x, pt_z, -pt_y])
            occupancy_map = self.point_to_map(points)
            cv2.imshow(f"Example occupancy_map{index}", occupancy_map)
            key = cv2.waitKey(1)
            occupancy_map_list.append(occupancy_map)
        self.observation_list.append(occupancy_map_list)


        for index in range(3,2*self.robot_num):
            q=Quaternion(argv[index].pose.pose.orientation.x,argv[index].pose.pose.orientation.y,argv[index].pose.pose.orientation.z,argv[index].pose.pose.orientation.w)
            pose_index=[argv[index].pose.pose.position.x,argv[index].pose.pose.position.y,q.to_euler(degrees=False)[0]]
            pose_list.append(pose_index)
        self.trace.append(pose_list)
        for index in range(0, self.robot_num):
            control_list.append(self.expert_control_gazebo(pose_list,index,True))

        for index in range(0,self.robot_num):
            msg=Twist()
            msg.linear.x = control_list[index][0] if abs(control_list[index][0])<0.1 else 0.1*abs(control_list[index][0])/control_list[index][0]
            msg.linear.y = control_list[index][1] if abs(control_list[index][1])<0.1 else 0.1*abs(control_list[index][1])/control_list[index][1]
            # msg.linear.x = 1
            # msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.z = 0.1
            self.pub_topic_dict[index].publish(msg)
        self.time_step+=1
        print(self.time_step)
        if self.time_step>self.max_time_step:
            self.save_to_file()
            rospy.signal_shutdown(f"Stop after {self.time_step} steps")





if __name__ == "__main__":


    state_msg = ModelState()
    state_msg.model_name = 'rm_0'
    state_msg.pose.position.x = 1
    state_msg.pose.position.y = 1
    state_msg.pose.position.z = 0

    q=Quaternion.from_euler(0, 0, 0, degrees=True)
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    resp = set_state(state_msg)
    state_msg = ModelState()
    state_msg.model_name = 'rm_1'
    state_msg.pose.position.x = 1
    state_msg.pose.position.y = -1
    state_msg.pose.position.z = 0
    q = Quaternion.from_euler(0, 0, 0, degrees=True)
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_state(state_msg)
    state_msg = ModelState()
    state_msg.model_name = 'rm_2'
    state_msg.pose.position.x = 0
    state_msg.pose.position.y = 0
    state_msg.pose.position.z = 0
    q = Quaternion.from_euler(0, 0, 0, degrees=True)
    state_msg.pose.orientation.x = q.x
    state_msg.pose.orientation.y = q.y
    state_msg.pose.orientation.z = q.z
    state_msg.pose.orientation.w = q.w
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_state(state_msg)
    rospy.init_node("collect_data")
    robot_num = 3
    listener = DataCollector(robot_num)
    rospy.spin()

