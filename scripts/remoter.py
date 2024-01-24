import socket
import threading
import time
import keyboard
class ControlPublisher():
    def __init__(self):
        self.count=0
        self.udp_socket= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_keyboard_message(self):
        if keyboard.is_pressed(hotkey='w'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[1,0,0];"
        elif keyboard.is_pressed(hotkey='a'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[0,1,0];"
        elif keyboard.is_pressed(hotkey='s'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[-1,0,0];"
        elif keyboard.is_pressed(hotkey='d'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[0,-2,0];"
        elif keyboard.is_pressed(hotkey='q'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[1,1,0];"
        elif keyboard.is_pressed(hotkey='e'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[1,-1,0];"
        elif keyboard.is_pressed(hotkey='z'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[1,-1,0];"
        elif keyboard.is_pressed(hotkey='x'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[-1,-1,0];"
        elif keyboard.is_pressed(hotkey='i'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[,0,-10];"
        elif keyboard.is_pressed(hotkey='r'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[0,0,-10];"
        elif keyboard.is_pressed(hotkey='t'):
            user_input = keyboard.read_event().name
            print(f"\nYou entered: {user_input}")
            message = "6:[0,0,10];"
        else:
            message="6:[0,0,0];"
        # print(message)
        broadcast_address = ('<broadcast>', 12345)
        self.udp_socket.sendto(message.encode(), broadcast_address)
        time.sleep(0.04)
    def send_broadcast_message(self):
        if self.count == 0:
            self.count = 1
            message="6:[1,0];"
        elif self.count == 1:
            self.count=2
            message = "6:[0,-1];"
        elif self.count == 2:
            self.count=3
            message = "6:[-1,0];"
        elif self.count == 3:
            self.count=4
            message = "6:[0,1];"
        elif self.count == 4:
            self.count = 4
            message = "6:[0,0];"
        print(message)
        broadcast_address = ('<broadcast>', 12345)
        self.udp_socket.sendto(message.encode(), broadcast_address)
        time.sleep(10)
        # self.udp_socket.close()
class ControlRec():
    def __init__(self):
        self.count=0
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', 12345))
        self.udp_socket.settimeout(1)
    def receive_broadcast_response(self):
        try:
            data, addr = self.udp_socket.recvfrom(1024)
            print(f"Received response from {addr}: {data.decode()}")
        except socket.timeout:
            print("No response received within the timeout period.")
        print(self.count)
        self.count+=1
        # self.udp_socket.close()


if __name__ == "__main__":
    # 启动发送广播消息的线程
    controller=ControlPublisher()
    # receiver=ControlRec()
    while True:
        sender_thread = threading.Thread(target=controller.send_keyboard_message)
        sender_thread.start()

        # 启动接收广播回应的线程
        # receiver_thread = threading.Thread(target=receiver.receive_broadcast_response)
        # receiver_thread.start()

        # 等待两个线程完成
        sender_thread.join()
        # receiver_thread.join()
        # time.sleep(0.4)