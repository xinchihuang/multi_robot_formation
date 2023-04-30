"""
A scene template
author: Xinchi Huang
"""

import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)
from collections import defaultdict
import numpy as np
from vrep import vrep_interface
from robot_template import Robot
from utils.gabreil_graph import get_gabreil_graph

from vrep.robot_executor_vrep import Executor
from vrep.robot_sensor_vrep import Sensor

from comm_data import SceneData
from recorder import Recorder
from controller_new import *
from model.LocalExpertController import LocalExpertController
class Scene:
    """
    Scene for multiple robots
    """

    def __init__(self):
        """
        robot_list: A list contains all robot in the scene
        []

        adjacency_list: A dict records robots' neighbor position and
        relative distance in gabreil graph
        {robot index:[(neighbor index, neighbor x, neighbor y,relative distance)..]..}

        client_id: A unique Id for the simulation environment
        """
        self.robot_list = []
        self.adjacency_list = defaultdict(list)
        self.position_list = None
        self.orientation_list = None
        self.client_id = vrep_interface.init_vrep()
        self.num_robot=5
        self.desired_distance=2.0
        self.initial_max_range=10
        self.initial_min_range=1
        self.platform = "vrep"
        self.model_path="/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit0.9.pth"
        self.controller=VitController(model_path=self.model_path,desired_distance=self.desired_distance)
        # self.controller = LocalExpertController(desired_distance=self.desired_distance)
        for i in range(self.num_robot):
            self.add_robot_vrep(i,controller=self.controller,desired_distance=self.desired_distance)
        self.reset_pose(self.initial_max_range, self.initial_min_range)



    def add_robot_vrep(self, robot_index,controller,desired_distance=1.0):
        """
        Add a robot in the scene
        :param robot_index: The robot index
        :return:
        """
        new_robot = Robot(
            sensor=Sensor(),
            executor=Executor(),
            controller=controller,
            platform=self.platform,

        )
        new_robot.index = robot_index
        new_robot.executor.initialize(robot_index, self.client_id)
        new_robot.sensor.client_id = self.client_id
        new_robot.sensor.robot_index = robot_index
        new_robot.sensor.robot_handle = new_robot.executor.robot_handle
        new_robot.sensor.get_sensor_data()
        self.robot_list.append(new_robot)


    def update_scene_data(self):
        """
        Update the adjacency list(Gabriel Graph) of the scene. Record relative distance

        """
        # print("Distance")
        node_num = len(self.robot_list)
        # collect robots' position in th scene
        position_list = []
        index_list = []
        orientation_list = []
        for i in range(node_num):
            index_list.append(self.robot_list[i].index)
            position = self.robot_list[i].sensor_data.position
            orientation = self.robot_list[i].sensor_data.orientation[-1]
            orientation_list.append(orientation)
            position_list.append(position)
        position_array = np.array(position_list)

        # Get Gabreil Graph
        gabriel_graph = get_gabreil_graph(position_array, node_num)

        # Create adjacency list
        new_adj_list = defaultdict(list)
        for i in range(node_num):
            for j in range(node_num):
                if gabriel_graph[i][j] == 1 and not i == j:
                    distance = (
                        (position_array[i][0] - position_array[j][0]) ** 2
                        + (position_array[i][1] - position_array[j][1]) ** 2
                    ) ** 0.5
                    new_adj_list[index_list[i]].append(
                        (
                            index_list[j],
                            position_array[j][0],
                            position_array[j][1],
                            distance,
                        )
                    )
        self.adjacency_list = new_adj_list
        self.position_list = position_list
        self.orientation_list = orientation_list
        # print("DISTANCE")
        # for r in self.adjacency_list:
        #     for n in self.adjacency_list[r]:
        #         print("edge:", r, n[0], "distance:", n[epoch5])

    def broadcast_all(self):
        """
        Send observations to all robots for GNN control
        Observations: (All robots' observation, adjacency_list)
        :return: None
        """
        output = SceneData()
        observation_list = []
        for robot in self.robot_list:
            # if robot.sensor_data==None:
            #     print("None")
            #     robot.get_sensor_data()
            observation = robot.sensor_data
            observation_list.append(observation)
        self.update_scene_data()

        output.observation_list = observation_list
        output.adjacency_list = self.adjacency_list
        output.position_list = self.position_list
        output.orientation_list = self.orientation_list
        for robot in self.robot_list:
            robot.scene_data = output

    def reset_pose(self, max_disp_range, min_disp_range):
        """
        Reset all robot poses in a circle
        :param max_disp_range: min distribute range
        :param min_disp_range: max distribute range


        pose_list:[[pos_x,pos_y,theta],[pos_x,pos_y,theta]]
        height: A default parameter for specific robot and simulator.
        Make sure the robot is not stuck in the ground
        """
        pose_list = [[-3, -3, 0], [-3, 3, 0], [3, 3, 0], [3, -3,0], [0, 0, 0],]
                     # [-5, 0, 0], [0, 5, 0], [5, 0, 0], [0, -2, 0]]
        num_robot = len(self.robot_list)

        for i in range(num_robot):
            pos_height = 0.1587
            position = [pose_list[i][0], pose_list[i][1], pos_height]

            orientation = [0, 0, pose_list[i][2]]

            robot_handle = self.robot_list[i].executor.robot_handle

            vrep_interface.post_robot_pose(
                self.client_id, robot_handle, position, orientation
            )



    ### sumilation related
    def simulate(self,max_simulation_time,time_step=0.05):
        simulation_time = 0
        data_recorder = Recorder()
        data_recorder.root_dir = "saved_data_test"

        while True:

            if simulation_time > max_simulation_time:
                break
            simulation_time += time_step
            print("robot control at time")
            print(simulation_time)

            for robot in self.robot_list:
                sensor_data = robot.get_sensor_data()
                control_data = robot.get_control_data()

                robot.execute_control()
                # record data
                data_recorder.record_sensor_data(sensor_data)
                data_recorder.record_robot_trace(sensor_data)
                data_recorder.record_controller_output(control_data)

            self.check_stop_condition()
            self.broadcast_all()
            vrep_interface.synchronize(self.client_id)
        data_recorder.save_to_file()
        # vrep_interface.stop(self.client_id)
        return 1
    def check_stop_condition(self):
        """
        :return:
        """
        if self.adjacency_list == None:

            return False
        else:
            for key, value in self.adjacency_list.items():
                for r in value:
                    print(
                        "distance between {r1:d} and {r2:d} is {r3:f}".format(
                            r1=key, r2=r[0], r3=r[3]
                        )
                    )

if __name__ == "__main__":
    simulate_scene=Scene()
    simulate_scene.simulate(50)