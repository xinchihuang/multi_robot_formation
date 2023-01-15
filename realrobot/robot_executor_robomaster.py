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




# 直连模式下，机器人默认 IP 地址为 192.168.2.1, 控制命令端口号为 40923
host = "192.168.2.1"
port = 40923


def main():

    address = (host, int(port))

    # 与机器人控制命令端口建立 TCP 连接
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Connecting...")

    s.connect(address)

    print("Connected!")

    while True:

        # 等待用户输入控制指令
        msg = input(">>> please input SDK cmd: ")

        # 当用户输入 Q 或 q 时，退出当前程序
        if msg.upper() == "Q":
            break

        # 添加结束符
        msg += ";"

        # 发送控制命令给机器人
        s.send(msg.encode("utf-8"))

        try:
            # 等待机器人返回执行结果
            buf = s.recv(1024)

            print(buf.decode("utf-8"))
        except socket.error as e:
            print("Error receiving :", e)
            sys.exit(1)
        if not len(buf):
            break

    # 关闭端口连接
    s.shutdown(socket.SHUT_WR)
    s.close()


if __name__ == "__main__":
    main()


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

    def execute_control(self, control_data):
        """
        Use interface/APIs to execute control in real world
        :param control_data: Controls to be execute
        """
        omega_left = control_data.omega_left
        omega_right = control_data.omega_right
        print("index", control_data.robot_index)
        print("left", omega_left)
        print("right", omega_right)
