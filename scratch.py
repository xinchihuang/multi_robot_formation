"""
author: Xinchi Huang
"""

from simulate import Simulation

test_simulation = Simulation(20, 0.05, 1)
test_simulation.initial_scene(5)
test_simulation.run()
