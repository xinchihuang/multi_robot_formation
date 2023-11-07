#!/usr/bin/env python3
import math
from utils.gabreil_graph import get_gabreil_graph_local,global_to_local
from comm_data import ControlData
from collections import defaultdict
from realrobots.robot_executor_robomaster import Executor
import socket
import argparse


class LocalExpertControllerRemote:
    def __init__(self,robot_id,desired_distance=1,sensor_range=5,K_f=1,max_speed=0.1,message_port=12345,command_port=40923):
        self.name = "LocalExpertControllerRemote"
        self.robot_id=robot_id
        self.desired_distance = desired_distance
        self.sensor_range = sensor_range
        self.K_f = K_f
        self.max_speed=max_speed
        self.message_socket=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.message_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.message_socket.bind(('', message_port))
        self.executor=Executor()

        # self.robot_ip_dict = defaultdict(str)
        # self.robot_ip_dict[0]="192.168.0.100"
        # self.robot_socket_pool=defaultdict()
        # for robot_id in self.robot_ip_dict:
        #     robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #     robot_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        #     robot_socket.connect((self.robot_ip_dict[robot_id], command_port))
        #     msg = "command"
        #     msg += ';'
        #     robot_socket.send(msg.encode('utf-8'))
        #     self.robot_socket_pool[robot_id] = robot_socket
        # print(self.robot_ip_dict)



    def remote_control(self):
        message, addr = self.message_socket.recvfrom(1024)  # Buffer size is 1024 bytes
        message=message.decode()
        print(message)
        try:
            marker_list= message.strip(";").split(";")
            pose_list=[]

            # for i in range(len(marker_list)):
            #     pose=[]
            #     pose_string_list=marker_list[i].strip("(").strip(")").split(",")
            #     for j in range(len(pose_string_list)):
            #         pose.append(float(pose_string_list[j]))
            #     temp = pose[1]
            #     pose[1] = -pose[2]
            #     pose[2] = 0
            #     print(i, pose)
            #     pose_list.append(pose)
            control_data = ControlData()

            for i in marker_list:
                id=marker_list[i].split(":")[0]
                control_x=marker_list[i].split(":")[1].strip('[').strip(']').split(" ")[0]
                control_y = marker_list[i].split(":")[1].strip('[').strip(']').split(" ")[1]
            if id==self.robot_id:
                control_data.velocity_x = control_x
                control_data.velocity_y = control_y
            # data={}
            # data["robot_id"] = self.robot_id
            # data["pose_list"] = pose_list
            # control_data = self.get_control(data)


            self.executor.execute_control(control_data=control_data)
        except:
            pass
        # for robot_id in range(len(pose_list)):
        #
        #     data["robot_id"]=robot_id
        #     data["pose_list"]=pose_list
        #     control_data=self.get_control(data)
        #     velocity_x = control_data.velocity_x
        #     velocity_y = control_data.velocity_y
        #
        #     omega = control_data.omega
        #     print("index", robot_id)
        #     print("x", velocity_x)
        #     print("y", velocity_y)
        #     print("omega", omega)
        #     velocity_x = 0.1 * abs(velocity_x) / velocity_x if abs(velocity_x) > 0.1 else velocity_x
        #     velocity_y = 0.1 * abs(velocity_y) / velocity_y if abs(velocity_y) > 0.1 else velocity_y
        #     omega = 0.1 * abs(omega) / omega if abs(omega) > 0.1 else omega
        #     # velocity_x=0
        #     # velocity_y=0
        #     # omega=1
        #     if velocity_x == 0 and velocity_y == 0 and omega == 0:
        #         msg = "chassis speed x {speed_x} y {speed_y} z {speed_z}".format(
        #             speed_x=velocity_x, speed_y=-velocity_y, speed_z=math.degrees(omega)
        #         )
        #     else:
        #         msg = "chassis speed x {speed_x} y {speed_y} z {speed_z}".format(
        #             speed_x=velocity_x, speed_y=-velocity_y, speed_z=math.degrees(omega)
        #         )
        #     msg += ';'
        #     self.robot_socket_pool[robot_id].send(msg.encode('utf-8'))



    def get_control(self,data):

        out_put = ControlData()
        try:
            self.robot_id = data["robot_id"]
            self.pose_list = data["pose_list"]
        except:
            print("Invalid input!")
            return out_put
        desired_distance = self.desired_distance
        gabreil_graph_local = get_gabreil_graph_local(self.pose_list, self.sensor_range,view_angle=math.pi*2 )
        pose_array_local= global_to_local(self.pose_list)
        neighbor_list = gabreil_graph_local[self.robot_id]
        velocity_sum_x = 0
        velocity_sum_y = 0
        for neighbor_id in range(len(neighbor_list)):
            if neighbor_id == self.robot_id or neighbor_list[neighbor_id] == 0:
                continue
            position_local=pose_array_local[self.robot_id][neighbor_id]
            distance_formation = (position_local[0] ** 2 + position_local[1] ** 2) ** 0.5
            rate_f = (distance_formation - desired_distance) / distance_formation
            velocity_x_f = rate_f * position_local[0]
            velocity_y_f = rate_f * position_local[1]
            velocity_sum_x += self.K_f*velocity_x_f
            velocity_sum_y += self.K_f*velocity_y_f
        # print(robot_id,velocity_x_f,velocity_x_l,velocity_x_r,velocity_x_s)
        out_put.velocity_x=velocity_sum_x if abs(velocity_sum_x)<self.max_speed else self.max_speed*abs(velocity_sum_x)/velocity_sum_x
        out_put.velocity_y=velocity_sum_y if abs(velocity_sum_y)<self.max_speed else self.max_speed*abs(velocity_sum_y)/velocity_sum_y
        return out_put
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-id', '--robot_id')
    args = parser.parse_args()
    controller=LocalExpertControllerRemote(int(args.robot_id))
    while True:
        controller.remote_control()
