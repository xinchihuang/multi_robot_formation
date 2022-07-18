"""
author: Xinchi Huang
"""

from simulate import Simulation
import numpy as np


test_simulation = Simulation(5, 0.05)
test_simulation.initial_scene(5)
test_simulation.run()
# point = np.load(
#     "/home/xinchi/multi_robot_formation/saved_data/0/0/points.npy", allow_pickle=True
# )
# print(len(point[1]))
