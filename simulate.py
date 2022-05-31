"""
A simulation template
author: Xinchi Huang
"""
from scene import scene
class simulation():
    def __init__(self,max_simulation_time,time_step,num_run,initial_range=5):
        self.max_simulation_time=max_simulation_time
        self.time_step=time_step
        self.num_run=num_run
        self.initial_range=initial_range
        self.client_id=None

    def run(self):
        for run_iter in range(self.num_run):
            pass
        return 1
    def initial_scene(self,num_robot):
        simulation_scene=scene(num_robot)
        simulation_scene.initial_vrep()
        for i in range(num_robot):
            simulation_scene.add_robot(i)
        simulation_scene.reset_positions(self.initial_range)
    def check_stop_condition(self):
        return True

