import numpy as np
import rospy
from realrobot.robot_executor_robomaster import EP,Executor
from comm_data import SceneData,SensorData
from collections import defaultdict
import time
from robot import Robot
class ModelControl:
    def __init__(self, topic):
        self.topic = topic
        # self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.ModelControlCallback)

        self.map_size = 100
        self.range = 5
        self.height = 2
        self.color_index = {"red":0,"yellow":1,"green":2}
        self.params = np.loadtxt("/home/xinchi/catkin_ws/src/localization/scripts/params.csv", delimiter=",")

        self.robot=Robot()
        self.executor=Executor()
        # self.executor.initialize()
        self.EP_DICT={}

        self.IP_DICT={0:'172.20.10.6',1:'172.20.10.7',2:'172.20.10.8'}
        # self.IP_DICT={1:'172.20.10.7'}

        for index,ip in self.IP_DICT.items():
            print('%s connecting...' % ip)
            self.EP_DICT[ip] = EP(ip)
            self.EP_DICT[ip].start()
    def ModelControlCallback(self, data):

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
    listener = ModelControl(topic)
    time.sleep(1)
    rospy.spin()


