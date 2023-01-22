#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""


import numpy as np
import socket
import sys
import math
from collections import defaultdict
import time
import threading
import serial
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


class UartConnector:

    def __init__(self):

        self.ser = serial.Serial()

        # 配置串口 波特率 115200，数据位 8 位，1 个停止位，无校验位，超时时间 0.2 秒
        self.ser.port = 'COM3'
        self.ser.baudrate = 115200
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.parity = serial.PARITY_NONE
        self.ser.timeout = 0.2
        # 打开串口
        self.ser.open()

        self.ser.write('command;'.encode('utf-8'))
    def send(self,msg):

        msg += ';'
        self.ser.write(msg.encode('utf-8'))
        recv = self.ser.readall()
        print(recv.decode('utf-8'))



class WifiConnector:
    def __init__(self):

        # 直连模式下，机器人默认 IP 地址为 192.168.2.1, 控制命令端口号为 40923
        self.host = "192.168.2.1"
        self.port = 40923
        self.address = (self.host, int(self.port))
        # 与机器人控制命令端口建立 TCP 连接
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.address)
        self.send_to_robot("command")

        try:
            # 等待机器人返回执行结果
            buf = self.s.recv(1024)

            print(buf.decode('utf-8'))
        except socket.error as e:
            print("Error receiving :", e)
            sys.exit(1)


    def send_to_robot(self,msg):

        # 添加结束符
        msg += ";"
        print(msg)
        # 发送控制命令给机器人
        self.s.send(msg.encode("utf-8"))
        try:
            # 等待机器人返回执行结果
            buf = self.s.recv(1024)

            print(buf.decode('utf-8'))
        except socket.error as e:
            print("Error receiving :", e)
            sys.exit(1)





class Executor:
    """
    A class to execute control from controller
    """

    def __init__(self):
        self.client_id = None
        self.robot_handle = None
        self.motor_left_handle = None
        self.motor_right_handle = None
        self.point_cloud_handle = None
        self.connector = WifiConnector()

    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in real world
        :param control_data: Controls to be executed
        """
        velocity_x = control_data.velocity_x*40
        velocity_y = control_data.velocity_y*40
        print("index", control_data.robot_index)
        print("left", velocity_x)
        print("right", velocity_y)
        # msg = "command"
        # self.connector.send_to_robot(msg)
        msg="chassis speed x {speed_x} y {speed_y} z {speed_z}".format(speed_x=velocity_x,speed_y=velocity_y,speed_z=10)
        self.connector.send_to_robot(msg)