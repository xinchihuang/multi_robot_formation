"""
A simulation template
author: Xinchi Huang
"""
from scene import scene


class simulation:
    def __init__(
        self,
        max_simulation_time,
        time_step,
        num_run,
        initial_max_range=5,
        initial_min_range=1,
    ):
        self.max_simulation_time = max_simulation_time
        self.time_step = time_step
        self.num_run = num_run
        self.initial_max_range = initial_max_range
        self.initial_min_range = initial_min_range
        self.client_id = None
        self.scene = None

    def run(self):
        while True:
            if self.check_stop_condition():
                break
            for robot in self.scene.robot_list:
                sensor_data = robot.get_sensor_data()
                print(sensor_data.position)
                control_data = robot.get_control_data()
                robot.execute_control()
        return 1

    def initial_scene(self, num_robot):
        simulation_scene = scene(num_robot)
        self.client_id = simulation_scene.initial_vrep()
        for i in range(num_robot):
            simulation_scene.add_robot(i)
        simulation_scene.reset_pose(self.initial_max_range, self.initial_min_range)
        self.scene = simulation_scene

    def check_stop_condition(self):
        return False
