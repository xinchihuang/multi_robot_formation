#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from cmvision.msg import Blobs
from cmvision_3d.msg import Blobs3d, Blob3d

import numpy as np
import socket
import sys
import math
from collections import defaultdict
import time
import threading

class EP:
    def __init__(self, ip):
        self._IP = ip
        self.__socket_ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket_isRelease = True
        self.__socket_isConnect = False
        self.__thread_ctrl_recv = threading.Thread(target=self.__ctrl_recv)
        self.__seq = 0
        self.__ack_list = []
        self.__ack_buf = 'ok'

    def __ctrl_recv(self):
        while self.__socket_isConnect and not self.__socket_isRelease:
            try:
                buf = self.__socket_ctrl.recv(1024).decode('utf-8')
                print('%s:%s' % (self._IP, buf))
                buf_list = buf.strip(";").split(' ')
                if 'seq' in buf_list:
                    print(buf_list[buf_list.index('seq') + 1])
                    self.__ack_list.append(int(buf_list[buf_list.index('seq') + 1]))
                self.__ack_buf = buf
            except socket.error as msg:
                print('ctrl %s: %s' % (self._IP, msg))

    def start(self):
        try:
            self.__socket_ctrl.connect((self._IP, 40923))
            self.__socket_isConnect = True
            self.__socket_isRelease = False
            self.__thread_ctrl_recv.start()
            self.command('command')
            self.command('robot mode free')
        except socket.error as msg:
            print('%s: %s' % (self._IP, msg))

    def exit(self):
        if self.__socket_isConnect and not self.__socket_isRelease:
            self.command('quit')
        self.__socket_isRelease = True
        try:
            self.__socket_ctrl.shutdown(socket.SHUT_RDWR)
            self.__socket_ctrl.close()
            self.__thread_ctrl_recv.join()
        except socket.error as msg:
            print('%s: %s' % (self._IP, msg))

    def command(self, cmd):
        self.__seq += 1
        cmd = cmd + ' seq %d;' % self.__seq
        print('%s:%s' % (self._IP, cmd))
        self.__socket_ctrl.send(cmd.encode('utf-8'))
        timeout = 2
        # while self.__seq not in self.__ack_list and timeout > 0:
        #     time.sleep(0.01)
        #     timeout -= 0.01
        if self.__seq in self.__ack_list:
            self.__ack_list.remove(self.__seq)
        return self.__ack_buf

class SceneData:
    """
    A class for passing data from scene
    """

    def __init__(self):
        self.observation_list = None
        self.adjacency_list = None

class SensorData:
    """
    A class for record sensor data
    """

    def __init__(self):
        self.robot_index = None
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.occupancy_map = None

class ControlData:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        self.omega_left = 0
        self.omega_right = 0


class Executor:
    """
    A class to execute control from controller
    """

    def __init__(self):
        self.socket = None
        self.host = "192.168.2.1"
        self.port = 40923
        self.address = (self.host, int(self.port))

    def initialize(self):
        # setup tcp connection with robot
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connecting...")
        self.socket.connect(self.address)
        print("Connected!")
        msg = "command"
        msg += ';'
        self.socket.send(msg.encode('utf-8'))

    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in real world
        :param control_data: Controls to be executed
        """
        speed_y = control_data.omega_left
        speed_x = control_data.omega_right
        print("index", control_data.robot_index)
        print("X", speed_x)
        print("Y", speed_y)

        # msg = "chassis wheel w2 "+ str(omega_left) +" w1 " +str(omega_right) +" w3 " +str(omega_right)+ " w4 " +str(omega_left)
        # msg += ';'
        msg="chassis speed x "+str(speed_x)+" y "+str(speed_y)+" z "+str(0)+";"
        print(msg)
        self.socket.send(msg.encode('utf-8'))
        # time.sleep(0.1)


def velocity_transform(velocity_x, velocity_y, theta):
        """
        Transform robot velocity to wheels velocity
        :param velocity_x:  robot velocity x (float)
        :param velocity_y: robot velocity y (float)
        :param theta: Robot orientation
        :return: wheel velocity left and right (float)
        """

        kk = 1
        max_velocity=1.2

        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        wheel_velocity_left = M11 * velocity_x + M12 * velocity_y
        wheel_velocity_right = M21 * velocity_x + M22 * velocity_y

        if (
            math.fabs(wheel_velocity_right) >= math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_right) > max_velocity
        ):
            alpha = max_velocity / math.fabs(wheel_velocity_right)
        elif (
            math.fabs(wheel_velocity_right) < math.fabs(wheel_velocity_left)
            and math.fabs(wheel_velocity_left) > max_velocity
        ):
            alpha = max_velocity / math.fabs(wheel_velocity_left)
        else:
            alpha = 1

        wheel_velocity_left = alpha * wheel_velocity_left
        wheel_velocity_right = alpha * wheel_velocity_right
        return wheel_velocity_left, wheel_velocity_right


def centralized_control(index, sensor_data, scene_data):
    """
    A centralized control, Expert control
    :param index: Robot index
    :param sensor_data: Data from robot sensor
    :param scene_data: Data from the scene
    :return: Control data
    """
    out_put = ControlData()
    desired_distance=1
    if not scene_data:
        out_put.omega_left = 0
        out_put.omega_right = 0
        return out_put
    self_robot_index = index

    self_position = sensor_data.position
    self_orientation = sensor_data.orientation
    self_x = self_position[0]
    self_y = self_position[1]
    print("self",self_x,self_y)
    neighbors = scene_data.adjacency_list[index]
    # print(neighbors)
    velocity_sum_x = 0
    velocity_sum_y = 0
    for neighbor in neighbors:
        rate = (neighbor[3] - desired_distance) / neighbor[3]
        velocity_x = rate * (self_x - neighbor[1])
        velocity_y = rate * (self_y - neighbor[2])
        velocity_sum_x -= velocity_x
        velocity_sum_y -= velocity_y
    print(velocity_sum_x)
    print(velocity_sum_y)
    if velocity_sum_x<-0.1:
        velocity_sum_x=-0.1
    elif velocity_sum_x>0.1:
        velocity_sum_x=0.1
    if velocity_sum_y<-0.1:
        velocity_sum_y=-0.1
    elif velocity_sum_y>0.1:
        velocity_sum_y=0.1
    # transform speed to wheels speed
    theta = sensor_data.orientation[2]
    out_put.robot_index = index
    out_put.omega_left = velocity_sum_x
    out_put.omega_right = velocity_sum_y
    return out_put

class ImageListener:
    def __init__(self, topic):
        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.imageDepthCallback)

        self.map_size = 1000
        self.range = 5
        self.height = 2
        self.color_index = {"red":0,"yellow":1,"green":2}
        self.params = np.loadtxt("/home/xinchi/catkin_ws/src/localization/scripts/params.csv", delimiter=",")
        self.executor=Executor()
        # self.executor.initialize()
        self.EP_DICT={}

        self.IP_DICT={0:'172.20.10.6',1:'172.20.10.7',2:'172.20.10.8'}
        # self.IP_DICT={1:'172.20.10.7'}

        for index,ip in self.IP_DICT.items():
            print('%s connecting...' % ip)
            self.EP_DICT[ip] = EP(ip)
            self.EP_DICT[ip].start()
    def imageDepthCallback(self, data):

        try:

            scene_data = SceneData()
            sensor_data_list=[None,None,None]
            position_dict={}
            adjacency_list= defaultdict(list)
            look_up_table=[0,0,0]
            for blob in data.blobs:
                if not blob.name in self.color_index:
                    continue

                robot_index=self.color_index[blob.name]
                if look_up_table[robot_index]==1:
                    continue
                look_up_table[robot_index]=1
                x_c, y_c, z_c = blob.center.x, blob.center.y, blob.center.z
                X_c_transpose = np.array([x_c, y_c, z_c, 1]).transpose()
                X_w_transpose = np.dot(self.params, X_c_transpose)
                x_w = X_w_transpose.transpose()[0]
                y_w = X_w_transpose.transpose()[1]
                z_w = X_w_transpose.transpose()[2]
                # print(blob.name,x_w,y_w,z_w)
                position_dict[robot_index]=[x_w,y_w,z_w]

                sensor_data = SensorData()
                sensor_data.position = [x_w,y_w,0]
                sensor_data.orientation=[0,0,0]
                sensor_data_list[robot_index]=sensor_data
            print(position_dict)
            for i in range(0,3):
                for j in range(0,3):
                    if i==j:
                        continue
                    distance = ((position_dict[i][0] - position_dict[j][0]) ** 2
                                       + (position_dict[i][1] - position_dict[j][1]) ** 2
                               ) ** 0.5
                    adjacency_list[i].append((j,position_dict[j][0],position_dict[j][1],distance))
            scene_data.adjacency_list=adjacency_list
            # print("AAAAAAAAAAAAA")

            for index, ip in self.IP_DICT.items():
                print(ip)
                control_data=centralized_control(index, sensor_data_list[index], scene_data)
                print(control_data.omega_left,control_data.omega_right)
                self.EP_DICT[ip].command('chassis speed x '+ str(control_data.omega_right)+' y '+str(control_data.omega_left)+' z 0')
                # self.EP_DICT[ip].command('chassis speed x 0 y 0 z 0')
            # self.executor.execute_control(control_data)


        except:

            return


if __name__ == '__main__':
    rospy.init_node("expert_control")
    topic = '/blobs_3d'
    listener = ImageListener(topic)
    time.sleep(1)
    rospy.spin()


