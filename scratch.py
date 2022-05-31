"""
author: Xinchi Huang
"""
import sim as vrep
import math
import random
from scene import scene
from simulate import simulation

test_simulation = simulation(20, 0.05, 1)
test_simulation.initial_scene(5)
test_simulation.run()
