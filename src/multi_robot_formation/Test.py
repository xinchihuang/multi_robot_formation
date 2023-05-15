"""
author: Xinchi Huang
"""

import os
import sys
sys.path.append("controller_new.py")
from simulate import Simulation
import numpy as np

# import torch
# a = torch.randn(2, epoch5, epoch1_6000)
#
# print(a)
# a=torch.permute(a, (2, 0, 1))
#
# print(a)
test_simulation = Simulation(50, 0.05)
test_simulation.initial_scene(5, "saved_model/vit0.9.pth")
test_simulation.run()
# point = np.load(
#     "/home/xinchi/multi_robot_formation/saved_data/0/0/points.npy", allow_pickle=True
# )
# print(len(point[1]))
