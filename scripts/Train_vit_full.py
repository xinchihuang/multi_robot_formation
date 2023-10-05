import os

print(os.getcwd())
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import copy
import math
import random
from model.vit_model import ViT
from utils.data_generator import DataGenerator
from utils.preprocess import preprocess

import cv2


class RobotDatasetTrace(Dataset):
    def __init__(
        self,
        data_path_root,
        desired_distance,
        number_of_agents,
        local,
        partial,
        max_x,
        max_y,
        sensor_view_angle,
        task_type="all",

    ):

        #### map simulator settings
        self.max_x = max_x
        self.max_y = max_y
        self.local = local
        self.partial = partial

        #### dataset settings
        self.transform = True
        self.desired_distance = desired_distance
        self.number_of_agents = number_of_agents
        self.task_type = task_type
        self.random_rate = 0
        self.num_sample = len(os.listdir(data_path_root))
        self.occupancy_maps_list = []
        self.pose_array = np.empty(shape=(self.number_of_agents, 1, 3))
        self.reference_control_list = []
        self.neighbor_list = []
        self.scale = np.zeros((self.number_of_agents, 1))
        for i in range(self.number_of_agents):
            self.scale[i, 0] = self.desired_distance
        for sample_index in tqdm(range(self.num_sample)):
            data_sample_path = os.path.join(data_path_root, str(sample_index))
            pose_array_i = np.load(
                os.path.join(data_sample_path, "trace.npy")
            )
            if self.pose_array.shape[1] == 1:
                self.pose_array = pose_array_i
                continue
            # print(pose_array_i.shape)
            self.pose_array = np.concatenate((self.pose_array, pose_array_i), axis=0)
        self.pose_array=np.transpose(self.pose_array,(1,0,2))


        self.sensor_view_angle = sensor_view_angle
        self.data_generator=DataGenerator(max_x=self.max_x,max_y=self.max_y,local=self.local, partial=self.partial,sensor_angle=self.sensor_view_angle)
        self.get_settings()

    def __len__(self):
        return self.pose_array.shape[1]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("pose array",idx,self.pose_array.shape)
        global_pose_array = self.pose_array[:, idx, :]
        self_orientation_array = global_pose_array[:, 2]
        self_orientation_array = copy.deepcopy(self_orientation_array)
        global_pose_array[:, 2] = 0

        occupancy_maps, reference_control, adjacency_lists,reference_position,reference_neighbor = self.data_generator.generate_map_all(
            global_pose_array
        )


        if self.transform:
            occupancy_maps = torch.from_numpy(occupancy_maps).double()
            reference_control = torch.from_numpy(reference_control).double()
            reference_position = torch.from_numpy(reference_position).double()
            reference_neighbor= torch.from_numpy(reference_neighbor).double()

        return {
            "occupancy_maps": occupancy_maps,
            "reference_control": reference_control,
            "reference_position": reference_position,
            "reference_neighbor": reference_neighbor,
        }
    def get_settings(self):
        print("-----------------------------------")
        print("Dataset settings")
        print("transform: ", self.transform)
        print("desired_distance: ", self.desired_distance)
        print("number_of_agents: ", self.number_of_agents)
        print("task_type: ", self.task_type)
        print("random_rate: ", self.random_rate)
        # print("random_range: ", self.random_range)
        print("num_sample: ", self.num_sample)
        print("data_shape:" , self.pose_array.shape)
        print("sensor_view_angle: ",self.sensor_view_angle)



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
        max_epoch=10,

        task_type="all",
        if_continue=False,
        load_model_path='',
        use_cuda=True,
        random_rate=1.,
        random_increase=True
    ):


        self.model = model.double()
        if if_continue:
            self.model.load_state_dict(
                torch.load(load_model_path, map_location=torch.device("cpu"))
            )
        self.trainset=trainset
        self.evaluateset=evaluateset

        self.number_of_agent = number_of_agent

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.task_type=task_type

        if criterion == "mse":
            self.criterion = nn.MSELoss()
        if optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(
                [p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate
            )
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.epoch = -1
        self.lr_schedule = {0: 0.0001, 10: 0.0001, 20: 0.0001}
        self.max_epoch=max_epoch

        self.random_rate = random_rate
        self.random_increase=random_increase
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.to("cuda")
        self.get_setting()
    def get_setting(self):
        print("-----------------------------------")
        print("Trainner settings")
        print("Transform: ", self.transform)
        print("Number_of_agents: ", self.number_of_agent)
        print("Task_type: ", self.task_type)
        print("Random_rate: ", self.random_rate)
        # print("random_range: ", self.random_range)
        print("batch_size: ", self.batch_size)
        print("learning_rate: ", self.learning_rate)
        print("use_cuda: ", self.use_cuda)
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

        while self.epoch < self.max_epoch:
            self.epoch += 1
            iteration = 0
            total_loss = 0
            total = 0

            # random_rate=0.1*self.epoch
            # trainloader.random_rate=random_rate
            for _, batch in enumerate(tqdm(trainloader)):
                occupancy_maps = batch["occupancy_maps"]
                reference = batch["reference_control"]
                if self.use_cuda:
                    occupancy_maps = occupancy_maps.to("cuda")
                    reference = reference.to("cuda")

                self.optimizer.zero_grad()
                outs = self.model(torch.unsqueeze(occupancy_maps[:,0,:,:],1),self.task_type)
                loss = self.criterion(outs, reference[:, 0])
                for i in range(1, self.number_of_agent):
                    outs = self.model(torch.unsqueeze(occupancy_maps[:,i,:,:],1),self.task_type)
                    loss += self.criterion(outs, reference[:, i])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total += occupancy_maps.size(0) * self.number_of_agent
                iteration += 1

                ### save and evaluating
                if iteration % 100 == 0:
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
                    self.save("/home/xinchi/vit_full/"+"model_" + str(iteration)+"_epoch"+str(self.epoch) + ".pth")
            self.save("/home/xinchi/vit_full/vit.pth")
        # return total_loss / total

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    # global parameters
    data_path_root = "/home/xinchi/gazebo_data"
    save_model_path = "/home/xinchi/vit_full/vit.pth"
    desired_distance = 2.0
    number_of_robot = 5
    map_size=100
    max_x = 5
    max_y =5
    sensor_view_angle= 2*math.pi
    # dataset parameters
    local = True
    partial = False

    #trainer parameters
    criterion = "mse"
    optimizer = "rms"
    batch_size = 128
    learning_rate= 0.01
    max_epoch=1
    use_cuda = True


    task_type="all"
    # model
    model=ViT(
        image_size = 100,
        patch_size = 10,
        num_classes = 3,
        dim = 256,
        depth = 3,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        agent_number=5
    )


    # data set
    trainset = RobotDatasetTrace(
        data_path_root=os.path.join(data_path_root, "training"),
        desired_distance=desired_distance,
        number_of_agents=number_of_robot,
        local=local,
        partial=partial,
        task_type=task_type,
        max_x=max_x,
        max_y=max_y,
        sensor_view_angle=sensor_view_angle
    )
    evaluateset = RobotDatasetTrace(
        data_path_root=os.path.join(data_path_root, "evaluating"),
        desired_distance=desired_distance,
        number_of_agents=number_of_robot,
        local=local,
        partial=partial,
        task_type=task_type,
        max_x=max_x,
        max_y=max_y,
        sensor_view_angle=sensor_view_angle
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
        max_epoch=max_epoch,
        use_cuda=use_cuda,
        task_type=task_type,
    )


    print(T.optimizer)
    T.train()


#

