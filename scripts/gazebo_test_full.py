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
from utils.initial_pose import initialize_pose,PoseDataLoader,initial_from_data
from utils.occupancy_map_simulator import MapSimulator

from controllers import VitController,LocalExpertControllerFull

class Simulation:
    def __init__(self, robot_num,controller,map_simulator,save_data_root=None,robot_upper_bound=0.12,robot_lower_bound=-0.12,sensor_range=5,max_velocity=0.05,max_simulation_time_step = 1000):

        # basic settings
        self.robot_num = robot_num
        self.occupancy_map_simulator = map_simulator
        # communication related
        self.sub_topic_list = []
        self.pub_topic_dict = collections.defaultdict()
        for index in range(self.robot_num):
            pose_topic=f'rm_{index}/odom'
            self.sub_topic_list.append(message_filters.Subscriber(pose_topic, Odometry))
        for index in range(self.robot_num):
            pub_topic=f'rm_{index}/cmd_vel'
            self.pub_topic_dict[index]=rospy.Publisher(pub_topic, Twist, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer(self.sub_topic_list, queue_size=10, slop=0.1,allow_headerless=True)
        ts.registerCallback(self.SimulateCallback)

        # robot related
        self.controller = controller
        self.robot_upper_bound=robot_upper_bound
        self.robot_lower_bound=robot_lower_bound
        self.sensor_range=sensor_range
        self.max_velocity=max_velocity

        # simulation related
        self.time_step=0
        self.max_simulation_time_step = max_simulation_time_step
        # save related
        self.save_data_root = save_data_root
        self.trace = []
        self.observation_list = []
        self.reference_control = []

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
            position_lists_local=global_to_local(pose_list)
            for index in range(0, self.robot_num):
                # print(position_lists_local[index])
                # start = time.time()
                occupancy_map = self.occupancy_map_simulator.generate_map_one(position_lists_local[index])
                # cv2.imshow("robot view " + str(index), np.array(occupancy_map))
                # cv2.waitKey(1)
                data={"robot_id":index,"pose_list":pose_list,"occupancy_map":occupancy_map}

                control_data = self.controller.get_control(data)
                # end=time.time()
                # print(end-start)
                control_list.append([control_data.velocity_x, control_data.velocity_y, control_data.omega])

        for index in range(0,self.robot_num):
            msg=Twist()
            # print(control_list[index])
            msg.linear.x = control_list[index][0] if abs(control_list[index][0])<self.max_velocity else self.max_velocity*abs(control_list[index][0])/control_list[index][0]
            msg.linear.y = control_list[index][1] if abs(control_list[index][1])<self.max_velocity else self.max_velocity*abs(control_list[index][1])/control_list[index][1]
            self.pub_topic_dict[index].publish(msg)
        self.time_step+=1

        if self.time_step>self.max_simulation_time_step:
            print("save")
            self.save_to_file()
            rospy.signal_shutdown(f"Stop after {self.time_step} steps")





if __name__ == "__main__":
    robot_num = 5
    # initial_pose="/home/xinchi/catkin_ws/src/multi_robot_formation/scripts/utils/poses_large_9"
    # # pose_lists=initial_from_data(initial_pose)
    # pose_list=pose_lists[random.randint(0,len(pose_lists)-1)]


    pose_list=initialize_pose(robot_num,initial_max_range=2)
    #
    # pose_list=[[0,0,0],
    #            # [1.5,1.5,0],
    #            # [-1.5,1.5,0],
    #            # [-1.5,-1.5,0],
    #            # [1.5,-1.5,0],
    #            [-1.5,0,0],
    #            [1.5,0,0],
    #            [0,1.5,0],
    #            [0,-1.5,0],
    #            # [3,0,0],
    #            # [-3,0,0],
    #            # [0,3,0],
    #            # [0,-3,0],
    #            ]


    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    rospy.init_node("collect_data")

    ### Vit controller
    model_path="/home/xinchi/catkin_ws/src/multi_robot_formation/scripts/saved_model/model_3200_epoch10.pth"
    save_data_root="/home/xinchi/gazebo_data/ViT_1m/ViT_13_1m"
    controller=VitController(model_path)
    #
    desired_distance = 1.0
    # sensor_range=5
    # K_f=1
    # max_speed = 1
    # controller = LocalExpertControllerFull(desired_distance=desired_distance,sensor_range=sensor_range,K_f=K_f,max_speed=max_speed)

 #
 #    pose_list=[[-1.8344854  ,-2.54902913  ,1.31531797],
 # [-0.11962687 ,-2.94522615 ,-2.78613711],
 # [-4.51360495  ,1.04370626  ,0.72373201],
 # [ 0.34727331  ,1.90429804 ,-1.54858546],
 # [-2.34736724  ,2.89713682 ,-1.14321162]]
    sensor_range=2
    sensor_view_angle = math.pi * 2
    occupancy_map_simulator = MapSimulator(max_x=sensor_range, max_y=sensor_range,
                                           sensor_view_angle=math.pi * 2, local=True, partial=False)
    listener = Simulation(robot_num=robot_num,controller=controller,map_simulator=occupancy_map_simulator,sensor_range=sensor_range,save_data_root=save_data_root)

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

