"""
author: Xinchi Huang
"""

from scene_vrep import Scene
from vrep.robot_executor_vrep import Executor
from vrep.robot_sensor_vrep import Sensor
from robot_template import Robot
from controller_new import *

### ViT experiments
num_robot=5
desired_distance=2.0
initial_max_range=5
initial_min_range=1
platform="vrep"
simulate_scene = Scene(num_robot=num_robot,desired_distance=desired_distance,initial_max_range=initial_max_range,initial_min_range=initial_min_range)
model_path="/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit1.0.pth"
controller = VitController(model_path=model_path,desired_distance=desired_distance)
for i in range(num_robot):
    new_robot = Robot(
        sensor=Sensor(),
        executor=Executor(),
        controller=controller,
        platform=platform,
    )
    simulate_scene.add_robot_vrep(i,new_robot)

simulate_scene.reset_pose(5, 1.5, 4)
simulate_scene.simulate(50, test_case="ViT")
simulate_scene.stop()