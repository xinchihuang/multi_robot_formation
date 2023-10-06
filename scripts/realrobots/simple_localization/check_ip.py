# -*- encoding: utf-8 -*-
import socket

ip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定 IP 广播端口
ip_sock.bind(('0.0.0.0', 40926))

# 等待接收数据
ip_str = ip_sock.recvfrom(1024)

# 输出数据
print(ip_str)