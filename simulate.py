"""
A simulation template
author: Xinchi Huang
"""
from scene import Scene
from recorder import Recorder
class Simulation:
    """
    Code for simulation
    """

    def __init__(
            self,
            max_simulation_time,
            time_step=0.05,
            initial_max_range=5,
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
        data_recorder=Recorder()
        data_recorder.root_dir="saved_data"
        while True:
            if self.check_stop_condition() or simulation_time > self.max_simulation_time:
                break
            simulation_time += self.time_step
            print("robot control at time")
            print(simulation_time)
            for robot in self.scene.robot_list:
                sensor_data=robot.get_sensor_data()
                control_data=robot.get_control_data()
                robot.execute_control()
                # record data
                data_recorder.record_sensor_data(sensor_data)
                data_recorder.record_robot_trace(sensor_data)
                data_recorder.record_controller_output(control_data)

            self.scene.update_adjacency_list()
            self.scene.broadcast_adjacency_list()
            break
        data_recorder.save_to_file()
        return 1

    def initial_scene(self, num_robot):
        """

        :param num_robot:
        :return:
        """
        simulation_scene = Scene()
        self.client_id = simulation_scene.initial_vrep()
        for i in range(num_robot):
            simulation_scene.add_robot(i)
        simulation_scene.reset_pose(self.initial_max_range, self.initial_min_range)

        self.scene = simulation_scene

    def check_stop_condition(self):
        """
        :return:
        """

        return False
