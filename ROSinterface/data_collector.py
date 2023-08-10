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

from multi_robot_formation.utils.gabreil_graph import get_gabreil_graph,get_gabreil_graph_local
from multi_robot_formation.utils.initial_pose import initialize_pose
class DataCollector:
    def __init__(self, robot_num):
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
        ts = message_filters.ApproximateTimeSynchronizer(self.sub_topic_list, queue_size=10, slop=0.1,allow_headerless=True)
        ts.registerCallback(self.DataCollectorCallback)

        self.save_data_root="/home/xinchi/gazebo_data"
        self.upper_bound=0.12
        self.lower_bound=-0.12
        self.map_size = 100
        self.range = 5
        self.height = 2
        self.max_time_step=1000

        self.sensor_range=5
        self.sensor_angle=120
        self.max_velocity=1
        self.max_omega=1

        self.desired_distance=2
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
    def expert_control_global(self,pose_list,robot_id):
        desired_distance=self.desired_distance
        gabreil_graph=get_gabreil_graph(pose_list)
        print("___________")
        neighbor_list=gabreil_graph[robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        for i in range(len(neighbor_list)):
            if i==robot_id or neighbor_list[i]==0:
                continue
            distance = ((pose_list[robot_id][0]-pose_list[i][0]) ** 2 + (pose_list[robot_id][1]-pose_list[i][1]) ** 2) ** 0.5
            print(robot_id,i,distance)
            rate = (distance - desired_distance) / distance
            velocity_x = rate * (pose_list[robot_id][0]-pose_list[i][0])
            velocity_y = rate * (pose_list[robot_id][1]-pose_list[i][1])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        return velocity_sum_x,velocity_sum_y
    def expert_control_local(self,pose_list,robot_id):
        desired_distance=self.desired_distance
        gabreil_graph_local=get_gabreil_graph_local(pose_list,self.sensor_range,self.sensor_angle)
        neighbor_list=gabreil_graph_local[robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        velocity_sum_omega=0
        robot_orientation = pose_list[robot_id][2]
        for i in range(len(neighbor_list)):
            if i==robot_id or neighbor_list[i]==0:
                continue
            distance = ((pose_list[robot_id][0]-pose_list[i][0]) ** 2 + (pose_list[robot_id][1]-pose_list[i][1]) ** 2) ** 0.5
            print(robot_id,i,distance)
            # if distance>self.sensor_range:
            #     continue
            # vector1 = np.array([pose_list[i][0]- pose_list[robot_id][0], pose_list[i][1]- pose_list[robot_id][1]])
            # vector2 = np.array([math.cos(robot_orientation), math.sin(robot_orientation)])
            # dot_product = np.dot(vector1, vector2)
            # magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            # inner_angle_rad = np.arccos(dot_product / magnitude_product)
            # inner_angle_deg = np.degrees(inner_angle_rad)
            # if inner_angle_deg > self.sensor_angle/2 :
            #     continue

            rate = (distance - desired_distance) / distance
            velocity_x = rate * (pose_list[robot_id][0]-pose_list[i][0])
            velocity_y = rate * (pose_list[robot_id][1]-pose_list[i][1])
            velocity_omega = robot_orientation-math.atan2((pose_list[i][1]-pose_list[robot_id][1]), (pose_list[i][0]- pose_list[robot_id][0]))
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
            velocity_sum_omega -= velocity_omega
        vx = velocity_sum_x * math.cos(robot_orientation) + velocity_sum_y * math.sin(robot_orientation)
        vy = -velocity_sum_x * math.sin(robot_orientation) + velocity_sum_y * math.cos(robot_orientation)
        # return velocity_sum_x,velocity_sum_y,velocity_sum_omega

        return vx, vy, velocity_sum_omega


    def DataCollectorCallback(self, *argv):
        # print(argv)
        occupancy_map_list = []
        pose_list = []
        control_list=[]
        # for index in range(0,self.robot_num):
        #     point_data=argv[index]
        #     points = []
        #     for point in point_cloud2.read_points(point_data, skip_nans=True):
        #         pt_x = point[0]
        #         pt_y = point[1]
        #         pt_z = point[2]
        #         if self.lower_bound< pt_y < self.upper_bound:
        #             # print([pt_x, pt_y, pt_z])
        #             points.append([pt_x, pt_z, -pt_y])
        #     occupancy_map = self.point_to_map(points)
        #     cv2.imshow(f"Example occupancy_map{index}", occupancy_map)
        #     key = cv2.waitKey(1)
        #     occupancy_map_list.append(occupancy_map)
        # self.observation_list.append(occupancy_map_list)
        for index in range(self.robot_num):
            q=Quaternion(argv[index].pose.pose.orientation.x,argv[index].pose.pose.orientation.y,argv[index].pose.pose.orientation.z,argv[index].pose.pose.orientation.w)
            pose_index=[argv[index].pose.pose.position.x,argv[index].pose.pose.position.y,q.to_euler(degrees=False)[0]]
            pose_list.append(pose_index)
        self.trace.append(pose_list)
        # print(self.trace)
        print("___________")
        for index in range(0, self.robot_num):
            control_list.append(self.expert_control_local(pose_list,index))
        for index in range(0,self.robot_num):
            msg=Twist()
            msg.linear.x = control_list[index][0] if abs(control_list[index][0])<self.max_velocity else self.max_velocity*abs(control_list[index][0])/control_list[index][0]
            msg.linear.y = control_list[index][1] if abs(control_list[index][1])<self.max_velocity else self.max_velocity*abs(control_list[index][1])/control_list[index][1]
            # msg.linear.x = 1
            # msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.z = control_list[index][2] if abs(control_list[index][2])<self.max_omega else self.max_omega*abs(control_list[index][2])/control_list[index][2]
            self.pub_topic_dict[index].publish(msg)
        self.time_step+=1

        if self.time_step>self.max_time_step:
            self.save_to_file()
            rospy.signal_shutdown(f"Stop after {self.time_step} steps")





if __name__ == "__main__":

    pose_list=initialize_pose(5)
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
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(state_msg)
    # state_msg = ModelState()
    # state_msg.model_name = 'rm_0'
    # state_msg.pose.position.x = 3
    # state_msg.pose.position.y = 3
    # state_msg.pose.position.z = 0
    # q=Quaternion.from_euler(0, 0, -135, degrees=True)
    # state_msg.pose.orientation.x = q.x
    # state_msg.pose.orientation.y = q.y
    # state_msg.pose.orientation.z = q.z
    # state_msg.pose.orientation.w = q.w
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # resp = set_state(state_msg)
    # state_msg = ModelState()
    # state_msg.model_name = 'rm_1'
    # state_msg.pose.position.x = -3
    # state_msg.pose.position.y = 3
    # state_msg.pose.position.z = 0
    # q = Quaternion.from_euler(0, 0, 0, degrees=True)
    # state_msg.pose.orientation.x = q.x
    # state_msg.pose.orientation.y = q.y
    # state_msg.pose.orientation.z = q.z
    # state_msg.pose.orientation.w = q.w
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # set_state(state_msg)
    # state_msg = ModelState()
    # state_msg.model_name = 'rm_2'
    # state_msg.pose.position.x = 0
    # state_msg.pose.position.y = 0
    # state_msg.pose.position.z = 0
    # q = Quaternion.from_euler(0, 0, 0, degrees=True)
    # state_msg.pose.orientation.x = q.x
    # state_msg.pose.orientation.y = q.y
    # state_msg.pose.orientation.z = q.z
    # state_msg.pose.orientation.w = q.w
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # set_state(state_msg)
    # state_msg = ModelState()
    # state_msg.model_name = 'rm_3'
    # state_msg.pose.position.x = 3
    # state_msg.pose.position.y = -3
    # state_msg.pose.position.z = 0
    # q = Quaternion.from_euler(0, 0, 135, degrees=True)
    # state_msg.pose.orientation.x = q.x
    # state_msg.pose.orientation.y = q.y
    # state_msg.pose.orientation.z = q.z
    # state_msg.pose.orientation.w = q.w
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # set_state(state_msg)
    # state_msg = ModelState()
    # state_msg.model_name = 'rm_4'
    # state_msg.pose.position.x = -3
    # state_msg.pose.position.y = -3
    # state_msg.pose.position.z = 0
    # q = Quaternion.from_euler(0, 0, 45, degrees=True)
    # state_msg.pose.orientation.x = q.x
    # state_msg.pose.orientation.y = q.y
    # state_msg.pose.orientation.z = q.z
    # state_msg.pose.orientation.w = q.w
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    # set_state(state_msg)

    rospy.init_node("collect_data")
    robot_num = 5
    listener = DataCollector(robot_num)
    rospy.spin()

