"""
A executor template. Record simulator/physical robot information,
 and used for execute control in simulator or real robot
author: Xinchi Huang
"""
import socket
import sys




class Executor:
    """
    A class to execute control from controller
    """

    def __init__(self,robot_index):
        self.socket=None
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
        omega_left = control_data.omega_left
        omega_right = control_data.omega_right
        print("index", control_data.robot_index)
        print("left", omega_left)
        print("right", omega_right)

        msg=f"chassis wheel w2 {omega_left} w1 {omega_right} w3 {omega_right} w4 {omega_left}"
        msg += ';'
        self.socket.send(msg.encode('utf-8'))
