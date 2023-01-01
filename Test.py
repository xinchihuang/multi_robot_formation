"""
author: Xinchi Huang
"""

from simulate import Simulation
import numpy as np

# import torch
# a = torch.randn(2, 3, 5)
#
# print(a)
# a=torch.permute(a, (2, 0, 1))
#
# print(a)
test_simulation = Simulation(1000, 0.05)
test_simulation.initial_scene(
    5, "/home/xinchi/multi_robot_formation/saved_model/model_final_expert.pth"
)
test_simulation.run()
# point = np.load(
#     "/home/xinchi/multi_robot_formation/saved_data/0/0/points.npy", allow_pickle=True
# )
# print(len(point[1]))
