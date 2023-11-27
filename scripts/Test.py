"""
author: Xinchi Huang
"""

from deprecated.scene_vrep_holonomic import SceneHolonomic
from deprecated.vrep import ExecutorHolonomic
from deprecated.vrep import SensorHolonomic
from deprecated.robot_template import Robot
from deprecated.controller_new import *
from model.LocalExpertController import LocalExpertControllerPartial

### ViT experiments
for i in range(1):
    simulation_time=50
    num_robot=5
    desired_distance=4.0
    initial_max_range=10
    initial_min_range=3
    max_sep_range=8
    sensor_range=10
    platform="vrep"
    simulate_scene = SceneHolonomic(num_robot=num_robot,desired_distance=desired_distance,initial_max_range=initial_max_range,initial_min_range=initial_min_range,max_sep_range=max_sep_range)
    model_path="/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit1.0.pth"
    # controller = VitController(model_path=model_path,desired_distance=desired_distance)
    controller = LocalExpertControllerPartial(desired_distance=desired_distance,view_range=sensor_range)
    for i in range(num_robot):
        new_robot = Robot(
            sensor=SensorHolonomic(),
            executor=ExecutorHolonomic(),
            controller=controller,
            platform=platform,
            sensor_range=sensor_range
        )
        simulate_scene.add_robot_vrep(i,new_robot)
    pose_list = [[-4, 0, math.pi/2],
                 [4, 0, -math.pi/2],
                 [-2, -2, 0],
                 [-2, 2, 0],
                 [2, 2, 0],
                 # [3, -3, 0],
                 # [0, 0, 0],
                ]
    simulate_scene.reset_pose()
    simulate_scene.simulate(simulation_time, test_case="Expert")
    simulate_scene.stop()



### ViT line formation experiments
# num_robot=7
# desired_distance=2.0
# initial_max_range=5
# initial_min_range=1
# sensor_range=10
# robot_platform="vrep"
# simulate_scene = Scene(num_robot=num_robot,desired_distance=desired_distance,initial_max_range=initial_max_range,initial_min_range=initial_min_range)
# model_path="/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit0.8.pth"
# controller = VitController(model_path=model_path,desired_distance=desired_distance)
# for i in range(num_robot):
#     if i==0 or i==1:
#         empty_controller=EmptyController()
#         new_robot = Robot(
#             sensor=Sensor(),
#             executor=Executor(),
#             controller=empty_controller,
#             robot_platform=robot_platform,
#             sensor_range=sensor_range
#         )
#     else:
#         new_robot = Robot(
#             sensor=Sensor(),
#             executor=Executor(),
#             controller=controller,
#             robot_platform=robot_platform,
#             sensor_range=sensor_range
#         )
#     simulate_scene.add_robot_vrep(i,new_robot)
# pose_list = [[-7, 0, 0],
#              [7, 0, 0],
#              [-4, -4, 0],
#              [-4, 4, 0],
#              [4, 4, 0],
#              [4, -4, 0],
#              [0, 0, 0],
#             ]
# simulate_scene.reset_pose(7, 1.5, 4,pose_list)
# simulate_scene.simulate(50, test_case="Line")
# simulate_scene.stop()