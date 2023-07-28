import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)

from comm_data import ControlData
import numpy as np
import math

class LocalExpertController:
    def __init__(self,desired_distance=2,safe_margin=0.5):
        self.desired_distance = desired_distance
        self.name="LocalExpertController"
        self.safe_margin=safe_margin
    def get_control(self,position_list_local):
        """
        :param position_list_local: local position list for training
        """
        position_array=np.array(position_list_local)
        out_put = ControlData()
        neighbor=np.ones(len(position_list_local))
        for v in range(len(position_list_local)):
            m = (position_array[v]) / 2
            for w in range(len(position_list_local)):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(m):
                    neighbor[v]=0
        velocity_sum_x =0
        velocity_sum_y =0
        num_neighbors=0
        for i in range(len(position_array)):
            # print(neighbor)
            if neighbor[i]==1:
                num_neighbors+=1
                if position_array[i][0]==float("inf") or position_array[i][1]==float("inf"):
                    continue
                distance = (position_array[i][0]** 2 + position_array[i][1]** 2)**0.5
                # print(position_array[i])
                # print(distance)
                rate = ((distance) - self.desired_distance) / (distance-self.safe_margin)
                velocity_x = rate * (-position_array[i][0])
                velocity_y = rate * (-position_array[i][1])
                velocity_sum_x -= velocity_x
                velocity_sum_y -= velocity_y
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y

        return out_put

class LocalExpertControllerPartial:
    def __init__(self,desired_distance=2,safe_margin=0.5,view_range=5,view_angle=120):
        self.desired_distance = desired_distance
        self.name="LocalExpertControllerPartial"
        self.safe_margin=safe_margin
        self.view_range=view_range
        self.view_angle=view_angle
    def get_control(self,position_list_local):
        """
        :param position_list_local: local position list for training
        """

        position_array=np.array(position_list_local)
        out_put = ControlData()
        neighbor=np.ones(len(position_list_local))

        for v in range(len(position_list_local)):
            if np.linalg.norm(position_array[v])>self.view_range:
                neighbor[v]=0
            if abs(math.atan2(position_array[v][1],position_array[v][0]))>self.view_angle/2:
                neighbor[v] = 0
            m = (position_array[v]) / 2
            for w in range(len(position_list_local)):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(m):
                    neighbor[v] = 0
        velocity_sum_x =0
        velocity_sum_y =0
        omerga_sum=0
        num_neighbors=0
        for i in range(len(position_array)):
            # print(neighbor)
            if neighbor[i]==1:
                num_neighbors+=1
                if position_array[i][0]==float("inf") or position_array[i][1]==float("inf"):
                    continue
                distance = (position_array[i][0]** 2 + position_array[i][1]** 2)**0.5
                # print(position_array[i])
                # print(distance)
                rate = ((distance) - self.desired_distance) / (distance-self.safe_margin)
                velocity_x = rate * (-position_array[i][0])
                velocity_y = rate * (-position_array[i][1])
                omega=-math.atan2(position_array[i][1],position_array[i][0])

                velocity_sum_x -= velocity_x
                velocity_sum_y -= velocity_y
                omerga_sum -= omega
        out_put.velocity_x = velocity_sum_x
        out_put.velocity_y = velocity_sum_y
        out_put.omega = omerga_sum
        return out_put
# if __name__ == "__main__":
#     pass
    # from utils.occupancy_map_simulator import MapSimulator
    # import cv2
    # import math
    #
    # pose_list=[[1.9944112300872803, -0.6254140138626099, 0.13870394229888916],
    #  [0.04576008766889572, 2.247776985168457, 0.13883700966835022],
    #  [2.7606122493743896, 1.425128698348999, 0.13866473734378815],
    #  [3.1266531944274902, -0.45182839035987854, 0.1386488378047943],
    #  [0.13337856531143188, 1.7551246881484985, 0.13878744840621948]]
    # orientation_list=[1.3544921875, 1.2664210796356201, 0.7540764212608337, -3.134005069732666, 2.650866985321045]
    # # pose_list = [[0.2 ,0, 0], [0, 0, 0]]
    # position_lists_global = pose_list
    # # orientation_list = [0,0,0,0,0]
    # occupancy_map_simulator = MapSimulator(local=True)
    # (
    #     position_lists_local,
    #     self_orientation,
    # ) = occupancy_map_simulator.global_to_local(
    #     np.array(position_lists_global), np.array(orientation_list)
    # )
    # print(position_lists_local[1])
    # occupancy_map = occupancy_map_simulator.generate_map_one(position_lists_local[1])
    #
    # controller=LocalExpertController()
    # out=controller.get_control(position_lists_local[1])
    # print(out.velocity_x,out.velocity_y)
    # cv2.imshow("robot view " + "(Synthesise)", occupancy_map)
    # cv2.waitKey(0)