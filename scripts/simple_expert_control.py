#!/usr/bin/env python3
import math
from utils.gabreil_graph import get_gabreil_graph_local,global_to_local
from comm_data import ControlData
from collections import defaultdict
from realrobots.robot_executor_robomaster import Executor
import socket
import argparse

import time
class ControlTransmitter:
    def __init__(self,robot_id,desired_distance=1,sensor_range=5,K_f=1,max_speed=0.1,message_port=12345):
        self.name = "ControlTransmitter"
        self.robot_id=robot_id
        self.desired_distance = desired_distance
        self.sensor_range = sensor_range
        self.K_f = K_f
        self.max_speed=max_speed
        self.message_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.message_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.message_socket.bind(('', message_port))
        self.executor=Executor()
        self.count=0
        self.input_data=None
    def __del__(self):
        print("Program Ends")
        control_data=ControlData()
        self.executor.execute_control(control_data)
    def get_input_data(self):
        try:
            message, addr = self.message_socket.recvfrom(1024)  # Buffer size is 1024 bytes
            message = message.decode()
            marker_list = message.strip(";").split(";")
            self.input_data=defaultdict(tuple)
            for i in range(len(marker_list)):
                object_id = int(marker_list[i].split(":")[0])
                control_x = float(marker_list[i].split(":")[1].strip('[').strip(']').split(",")[0])
                control_y = float(marker_list[i].split(":")[1].strip('[').strip(']').split(",")[1])
                self.input_data[object_id]=(control_x,control_y)
        except:
            pass
    def get_control(self):
        control_data = ControlData()
        try:
            self.get_input_data()
            control_x = self.input_data[self.robot_id][0]
            control_y = self.input_data[self.robot_id][1]
            control_data.velocity_x = control_x
            control_data.velocity_y = control_y
            control_data.robot_index = self.robot_id
        except:
            pass
        return control_data
    def excute_control(self):
        control_data=self.get_control()
        self.executor.execute_control(control_data)
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-id', '--robot_id')
    args = parser.parse_args()
    controller=ControlTransmitter(int(args.robot_id))
    while True:
        controller.excute_control()
