"""
author: Xinchi Huang
"""
import torch

from simulate import Simulation
import numpy as np
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.GNN_based_model import DecentralController, DecentralControllerPose
from model.Train_GNN_pose import RobotDatasetTrace
from utils.data_generator import DataGenerator
import cv2

# model=DecentralController(number_of_agent=5)
model = DecentralControllerPose(number_of_agent=5)
model.load_state_dict(
    torch.load("/home/xinchi/multi_robot_formation/saved_model/model_12000.pth")
)
dataset = RobotDatasetTrace(data_path_root="/home/xinchi/gnn_data/expert_adjusted_5")
testloader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
#
# global_pose_array=[[-0.76168436,  3.76078224,  0.68687618],
#  [ 1.20477724,  0.83772224,  1.27681875],
#  [ 3.92995429,  0.77855873,  0.76791084],
#  [-0.33735606,  0.86489499,  1.10176027],
#  [-1.24217212, -0.91493171,  3.01800156]]
# self_orientation_array=[0.68687618, 1.27681875, 0.76791084, 1.10176027, 3.01800156]
# global_pose_array=np.array(global_pose_array)
# global_pose_array[:, 2] = 0
# occupancy_maps, reference, adjacency_lists = generate_one(
#     global_pose_array, self_orientation_array, 2
# )
#
# print(global_pose_array)
# print(self_orientation_array)
# neighbor = np.zeros((1,5, 5))
# for key, value in adjacency_lists[0].items():
#     for n in value:
#         neighbor[0][key][n[0]] = 1
# refs = np.zeros((1,5, 1))
# scale = np.zeros((1,5, 1))
# scale[:,:,0]=2
# occupancy_maps=np.expand_dims(occupancy_maps,axis=0)
# occupancy_maps = torch.from_numpy(occupancy_maps).double()
#
#
#
# reference = torch.from_numpy(reference).double()
# neighbor = torch.from_numpy(neighbor).double()
# useless = torch.from_numpy(refs).double()
# scale = torch.from_numpy(scale).double()
#
# print(occupancy_maps.shape)
# # for i in range(occupancy_maps.shape[1]):
# #     cv2.imshow(str(i), occupancy_maps.numpy()[0][i])
# #     cv2.waitKey(0)
# print(neighbor.shape)
# print(scale.shape)
# print(useless.shape)
#
# model.addGSO(neighbor)
# outs = model(occupancy_maps, useless, scale)
# # for i in range(len(outs)):
# #     print(outs[i][0],reference[i])
# import torch.nn as nn
# criterion=nn.MSELoss()
# loss = criterion(outs[0][0], reference[0])
# print(outs[0][0],reference[0])
# for i in range(1, 5):
#     loss += criterion(outs[i][0], reference[i])
#     print(outs[i][0],reference[i])
#
# print(loss)

import torch.nn as nn

for iter, batch in enumerate(testloader):
    position_lists_local = batch["position_lists_local"]
    self_orientation = batch["self_orientation"]
    neighbor = batch["neighbor"]
    reference = batch["reference"]
    useless = batch["useless"]
    scale = batch["scale"]
    model.addGSO(neighbor)
    print(position_lists_local.shape)
    print(neighbor.shape)
    print(scale.shape)
    print(useless.shape)
    outs = model(position_lists_local, useless, scale)
    # print(outs)
    # print(reference)
    criterion = nn.MSELoss()
    loss = criterion(outs[0][0], reference[0][0])
    print(outs[0][0], reference[0][0])
    for i in range(1, 5):
        loss += criterion(outs[i][0], reference[0][i])
        print(outs[i][0], reference[0][i])

    print(loss)
    break
