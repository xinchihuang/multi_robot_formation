"""
author: Xinchi Huang
"""

import os
import sys
sys.path.append("model")

from simulate import Simulation
import numpy as np

# import torch
# a = torch.randn(2, 3, 5)
#
# print(a)
# a=torch.permute(a, (2, 0, 1))
#
# print(a)
test_simulation = Simulation(50, 0.05)
test_simulation.initial_scene(5, "saved_model/model_map_local_partial.pth")
test_simulation.run()
# point = np.load(
#     "/home/xinchi/multi_robot_formation/saved_data/0/0/points.npy", allow_pickle=True
# )
# print(len(point[1]))
