
import os
current_working_directory = os.getcwd()
from plots.plot_scene import plot_load_data_gazebo
print(current_working_directory)
root_path = "/home/xinchi/gazebo_data/ViT_1m/ViT_5_1m"
for path in os.listdir(root_path):
    plot_load_data_gazebo(os.path.join(root_path, path))