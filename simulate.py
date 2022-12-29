"""
A simulation template
author: Xinchi Huang
"""
import time

from scene import Scene
from recorder import Recorder
from vrep import vrep_interface


class Simulation:
    """
    Code for simulation
    """

    def __init__(
        self,
        max_simulation_time,
        time_step=0.05,
        initial_max_range=10,
        initial_min_range=1,
    ):
        self.max_simulation_time = max_simulation_time
        # Simulator related parameter. Used only for record and check stopping condition
        self.time_step = time_step

        self.initial_max_range = initial_max_range
        self.initial_min_range = initial_min_range
        self.client_id = None
        self.scene = None

    def run(self):
        """

        :return:
        """
        simulation_time = 0
        data_recorder = Recorder()
        data_recorder.root_dir = "saved_data"

        while True:
            # vrep_interface.synchronize(self.client_id)
            time.sleep(0.2)
            if (
                self.check_stop_condition()
                or simulation_time > self.max_simulation_time
            ):
                break
            simulation_time += self.time_step
            print("robot control at time")
            print(simulation_time)
            self.scene.broadcast_all()
            for robot in self.scene.robot_list:
                # print("robot index",robot.index)
                sensor_data = robot.get_sensor_data()
                control_data = robot.get_control_data()
                robot.execute_control()
                # record data
                data_recorder.record_sensor_data(sensor_data)
                data_recorder.record_robot_trace(sensor_data)
                data_recorder.record_controller_output(control_data)

        data_recorder.save_to_file()
        # vrep_interface.stop(self.client_id)
        return 1

    def initial_scene(self, num_robot, model_path):
        """

        :param num_robot:
        :return:
        """
        simulation_scene = Scene()
        self.client_id = simulation_scene.initial_vrep()
        for i in range(num_robot):
            simulation_scene.add_robot_vrep(i)
        simulation_scene.reset_pose(self.initial_max_range, self.initial_min_range)
        simulation_scene.initial_GNN(num_robot, model_path)
        self.scene = simulation_scene

    def check_stop_condition(self):
        """
        :return:
        """

        return False
