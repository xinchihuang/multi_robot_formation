import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn.functional as F
from utils.data_generator import DataGenerator
import math
import random
from vit_pytorch import ViT


class RobotDatasetTrace(Dataset):
    def __init__(
        self,
        data_path_root,
        desired_distance,
        number_of_agents,
        local,
        partial,
    ):

        self.transform = True
        self.local = local
        self.partial = partial
        self.desired_distance = desired_distance
        self.number_of_agents = number_of_agents

        self.num_sample = len(os.listdir(data_path_root))
        self.occupancy_maps_list = []
        self.pose_array = np.empty(shape=(5, 1, 3))
        self.reference_control_list = []
        self.neighbor_list = []
        self.scale = np.zeros((self.number_of_agents, 1))
        for i in range(self.number_of_agents):
            self.scale[i, 0] = self.desired_distance
        for sample_index in tqdm(range(self.num_sample)):
            data_sample_path = os.path.join(data_path_root, str(sample_index))
            pose_array_i = np.load(
                os.path.join(data_sample_path, "pose_array_scene.npy")
            )
            if self.pose_array.shape[1] == 1:
                self.pose_array = pose_array_i
                continue
            self.pose_array = np.concatenate((self.pose_array, pose_array_i), axis=1)
        print(self.pose_array.shape)

    def __len__(self):
        return self.pose_array.shape[1]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        global_pose_array = self.pose_array[:, idx, :]

        self_orientation_array = global_pose_array[:, 2]
        self_orientation_array = np.copy(self_orientation_array)
        global_pose_array[:, 2] = 0

        use_random = random.uniform(0, 1)
        if use_random > 0.9:
            global_pose_array = 2 * np.random.random((self.number_of_agents, 3)) - 1
            global_pose_array[:, 2] = 0
            self_orientation_array = (
                    2 * math.pi * np.random.random(self.number_of_agents) - math.pi
            )

        data_generator = DataGenerator(local=self.local, partial=self.partial)
        occupancy_maps, reference, adjacency_lists = data_generator.generate_one(
            global_pose_array, self_orientation_array
        )

        # for i in range(0,5):
        #     print(reference[i])
        # #     print(global_pose_array[i])
        #     cv2.imshow(str(i), occupancy_maps[i])
        #     cv2.waitKey(0)

        neighbor = np.zeros((self.number_of_agents, self.number_of_agents))
        for key, value in adjacency_lists[0].items():
            for n in value:
                neighbor[key][n[0]] = 1
        refs = np.zeros((self.number_of_agents, 1))

        scale = self.scale

        if self.transform:
            occupancy_maps = torch.from_numpy(occupancy_maps).double()
            reference = torch.from_numpy(reference).double()
            neighbor = torch.from_numpy(neighbor).double()
            refs = torch.from_numpy(refs).double()
            scale = torch.from_numpy(scale).double()
        return {
            "occupancy_maps": occupancy_maps,
            "neighbor": neighbor,
            "reference": reference,
            "useless": refs,
            "scale": scale,
        }



class Trainer:
    def __init__(
        self,
        model,
        trainset,
        evaluateset,
        number_of_agent,
        criterion,
        optimizer,
        batch_size,
        learning_rate,
        use_cuda,
    ):
        self.model = model.double()
        self.trainset=trainset
        self.evaluateset=evaluateset

        self.number_of_agent = number_of_agent

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        print(criterion)
        if criterion == "mse":
            self.criterion = nn.MSELoss()
        print(optimizer)
        if optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(
                [p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate
            )
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.epoch = -1
        self.lr_schedule = {0: 0.0001, 10: 0.0001, 20: 0.0001}

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.to("cuda")

    def train(self):
        """ """
        self.epoch += 1
        if self.epoch in self.lr_schedule.keys():
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr_schedule[self.epoch]

        trainloader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        evaluateloader = DataLoader(
            self.evaluateset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.model.train()
        total_loss = 0
        total = 0
        while self.epoch < 1:
            self.epoch += 1
            iteration = 0
            for iter, batch in enumerate(tqdm(trainloader)):
                occupancy_maps = batch["occupancy_maps"]
                reference = batch["reference"]
                if self.use_cuda:
                    occupancy_maps = occupancy_maps.to("cuda")
                    reference = reference.to("cuda")
                self.optimizer.zero_grad()
                print(occupancy_maps.shape)
                outs = self.model(torch.unsqueeze(occupancy_maps[:,0,:,:],1))
                loss = self.criterion(outs, reference[:, 0])

                for i in range(1, self.number_of_agent):
                    outs = self.model(torch.unsqueeze(occupancy_maps[:,i,:,:],1))
                    loss += self.criterion(outs, reference[:, i])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total += occupancy_maps.size(0) * self.number_of_agent
                iteration += 1
                if iteration % 1000 == 0:

                    print(
                        "Average training_data loss at iteration "
                        + str(iteration)
                        + ":",
                        total_loss / total,
                    )
                    print("loss ", total_loss)
                    print("total ", total)
                    total_loss = 0
                    total = 0
                    self.save("model_" + str(iteration) + ".pth")
                    evaluate(
                        evaluateloader,
                        self.use_cuda,
                        self.model,
                        self.optimizer,
                        self.criterion,
                        self.number_of_agent,
                        iteration,
                    )

        return total_loss / total

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)


def evaluate(evaluateloader, use_cuda, model, optimizer, criterion, nA, iteration):
    total_loss_eval = 0
    total_eval = 0
    for iter_eval, batch_eval in enumerate(evaluateloader):
        occupancy_maps = batch_eval["occupancy_maps"]
        neighbor = batch_eval["neighbor"]
        reference = batch_eval["reference"]
        useless = batch_eval["useless"]
        scale = batch_eval["scale"]
        if use_cuda:
            occupancy_maps = occupancy_maps.to("cuda")
            neighbor = neighbor.to("cuda")
            reference = reference.to("cuda")
            useless = useless.to("cuda")
            scale = scale.to("cuda")
        model.addGSO(neighbor)
        optimizer.zero_grad()
        # print(occupancy_maps.shape)

        outs = model(occupancy_maps, {},0,useless, scale)
        loss = criterion(outs[0], reference[:, 0])

        for i in range(1, nA):
            loss += criterion(outs[i], reference[:, i])
        optimizer.step()
        total_loss_eval += loss.item()
        total_eval += occupancy_maps.size(0) * nA
    print(
        "Average evaluating_data loss at iteration " + str(iteration) + ":",
        total_loss_eval / total_eval,
    )
    print("loss_eval ", total_loss_eval)
    print("total_eval ", total_eval)


if __name__ == "__main__":
    # global parameters
    data_path_root = "/home/xinchi/GNN_data"
    save_model_path = "/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/vit.pth"
    desired_distance = 2.0
    number_of_robot = 5
    map_size=100

    # dataset parameters
    local = True
    partial = False

    #trainer parameters
    criterion = "mse"
    optimizer = "rms"
    batch_size = 16
    learning_rate= 0.01
    use_cuda = True

    # model
    model=ViT(
        image_size = 100,
        patch_size = 10,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )


    # data set
    trainset = RobotDatasetTrace(
        data_path_root=os.path.join(data_path_root, "training"),
        desired_distance=desired_distance,
        number_of_agents=number_of_robot,
        local=local,
        partial=partial,
    )
    evaluateset = RobotDatasetTrace(
        data_path_root=os.path.join(data_path_root, "evaluating"),
        desired_distance=desired_distance,
        number_of_agents=number_of_robot,
        local=local,
        partial=partial,
    )



    T = Trainer(
        model=model,
        trainset=trainset,
        evaluateset=evaluateset,
        number_of_agent=number_of_robot,
        criterion=criterion,
        optimizer=optimizer,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_cuda=use_cuda,
    )
    print(T.optimizer)
    T.train()
    T.save(save_model_path)

#
