#!/usr/bin/env python3
import math
import rospy
import os
import cv2
import numpy as np
from realrobots.robot_executor_robomaster import Executor
from comm_data import SceneData, SensorData,ControlData
from controllers import VitController
from utils.occupancy_map_simulator import MapSimulator

from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud
from sensor_msgs import point_cloud2
from cmvision.msg import Blobs
from cmvision_3d.msg import Blobs3d, Blob3d
def find_connected_components_with_count(matrix):
    def dfs(r, c, component_number):
        if r < 0 or r >= rows or c < 0 or c >= cols or matrix[r][c] != 1:
            return 0

        matrix[r][c] = component_number
        count = 1  # Count the current cell

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1),(1,1),(1,-1),(-1,1),(-1,-1),
                       (2, 0), (-2, 0), (0, 2), (0, -2),(2,2),(2,-2),(-2,2),(-2,-2),
                       (3, 0), (-3, 0), (0, 3), (0, -3),(3,3),(3,-3),(-3,3),(-3,-3),]:
            count += dfs(r + dr, c + dc, component_number)

        return count

    rows = matrix.shape[0]
    cols = matrix.shape[1]
    component_number = 2  # Start component numbering from 2
    component_counts = {}  # Dictionary to store counts for each component

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                count = dfs(r, c, component_number)
                component_counts[component_number] = count
                component_number += 1

    return matrix, component_counts
def check_valid_components(matrix,component_counts,upper=10,lower=5):
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    center_dict={}

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]==0:
                continue
            component_id=matrix[i][j]
            # print(component_id)
            component_count=component_counts[component_id]
            if component_count<lower or component_count>upper:
                pass
            else:
                if not component_id in center_dict:
                    center_dict[component_id]=[0,0]
                center_dict[component_id][0] = center_dict[component_id][0] + i
                center_dict[component_id][1] = center_dict[component_id][1] + j
    for component_id in center_dict:
        center_dict[component_id][0] = int(center_dict[component_id][0] / component_counts[component_id])
        center_dict[component_id][1] = int(center_dict[component_id][1] / component_counts[component_id])

    return matrix,center_dict

class ModelControl:
    def __init__(self, topic):

        self.model_path = os.path.abspath(
            '') + "/catkin_ws/src/multi_robot_formation/scripts/saved_model/model_15600_epoch1.pth"
        self.desired_distance=1.0
        self.controller=VitController(model_path=self.model_path,desired_distance=self.desired_distance)
        # self.controller=LocalExpertController()

        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, PointCloud, self.ModelControlCallback)
        self.map_size = 100
        self.sensor_range = 2
        self.robot_size=0.15
        self.executor=Executor()
        self.color_index = {"Green": 0}
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
        point_map=np.zeros((self.map_size,self.map_size))
        scale = self.map_size
        robot_range = max(1, int(math.floor(self.map_size * self.robot_size / scale)))
        occupancy_map = (
                np.ones((self.map_size + 2 * robot_range, self.map_size + 2 * robot_range)) * 255
        )
        for point in data.points:
            if abs(point.x)>self.sensor_range or abs(point.y)>self.sensor_range:
                continue
            # print(point)
            x_world=-point.y
            y_world=point.x
            # print(x_world,y_world)
            y_map = min(int(self.map_size / 2) + int(x_world * self.map_size / self.sensor_range / 2), self.map_size - 1)
            x_map = min(int(self.map_size / 2) - int(y_world * self.map_size / self.sensor_range / 2), self.map_size - 1)
            if 0 <= x_map < self.map_size and 0 <= y_map < self.map_size:
                point_map[x_map][y_map]=1
                # print(x_world,y_world)
        connected_components, component_counts = find_connected_components_with_count(point_map)
        _,center_dict=check_valid_components(connected_components,component_counts)
        # print(component_counts)
        # print(center_dict)

        for object in center_dict:
            # print(center_dict[object])
            x=center_dict[object][0]
            y=center_dict[object][1]
            for m in range(-robot_range, robot_range, 1):
                for n in range(-robot_range, robot_range, 1):
                    occupancy_map[x + m][y + n] = 0

        occupancy_map = occupancy_map[
                        robot_range:-robot_range, robot_range:-robot_range
                        ]
        # for component_number, count in component_counts.items():
        #     print(f"Component {component_number}: {count} '1's")
        cv2.imshow("robot view " + str(0), np.array(occupancy_map))
        cv2.waitKey(1)
        cv2.imshow("raw" + str(0), point_map)
        cv2.waitKey(1)
        cv2.imwrite("/home/xinchi/raw.png",point_map)
        # data = {"robot_id": 0, "occupancy_map": occupancy_map}
        # control_data = self.controller.get_control(data)
        # self.executor.execute_control(control_data=control_data)
def stop_node(event):
    rospy.signal_shutdown("Time's up!")
if __name__ == "__main__":
    # signal.signal(signal.SIGINT, handler)
    rospy.init_node("model_control")
    topic = "pointcloud2d"
    listener = ModelControl(topic)
    # rospy.Subscriber('keyboard_input', String, listener.keyboard_stop)
    # timer = rospy.Timer(rospy.Duration(100), listener.timed_stop)
    rospy.spin()

