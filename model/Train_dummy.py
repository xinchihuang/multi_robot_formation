import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from model.GNN_based_model import DummyModel
import os
from utils.data_generator import DataGenerator
import math
import random
import cv2
class RobotDatasetTrace(Dataset):
    def __init__(self, data_path_root):

        self.transform = True
        self.desired_distance = 2
        self.number_of_agents = 5

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

        use_random=random.choice([True,False])
        # print(print("use random data"),use_random)
        if use_random:
            global_pose_array=10*np.random.random((self.number_of_agents,3))-5
            global_pose_array[:,2]=0
            self_orientation_array=2*math.pi*np.random.random(self.number_of_agents)-math.pi

        # global_pose_array=[[-4, -4, 0], [-4, 4, 0], [4, 4, 0], [4, -4, 0], [0, 0, 0]]
        # self_orientation_array=[math.pi/3,0,0,0,0]
        data_generator=DataGenerator(local=True)
        position_lists_local,self_orientation, reference, adjacency_lists = data_generator.generate_pose_one(global_pose_array, self_orientation_array)
        # print(position_lists_local[0])
        # print(reference[0])
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
            position_lists_local=torch.from_numpy(position_lists_local).double()
            self_orientation=torch.from_numpy(self_orientation).double()
            reference = torch.from_numpy(reference).double()
            neighbor = torch.from_numpy(neighbor).double()
            refs = torch.from_numpy(refs).double()
            scale = torch.from_numpy(scale).double()
        return {
            "position_lists_local": position_lists_local,
            "self_orientation":self_orientation,
            "neighbor": neighbor,
            "reference": reference,
            "useless": refs,
            "scale": scale,
        }


class Trainer:
    def __init__(
        self,
        criterion="mse",
        optimizer="rms",
        batch_size=128,
        number_of_agent=5,
        lr=0.01,
        cuda=True,
        if_continue=False,
        load_model_path=None
    ):
        self.points_per_ep = None
        self.number_of_agent = number_of_agent
        self.batch_size = batch_size
        self.model = DummyModel(
            number_of_agent=self.number_of_agent,
            use_cuda=cuda,
        ).double()
        self.if_continue=if_continue
        if self.if_continue:
            self.model.load_state_dict(torch.load(load_model_path))

        self.use_cuda = cuda
        if self.use_cuda:
            self.model = self.model.to("cuda")
        self.lr = lr
        if criterion == "mse":
            self.criterion = nn.MSELoss()
        if optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(
                [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
            )
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.epoch = -1
        self.lr_schedule = {0: 0.0001, 10: 0.0001, 20: 0.0001}
        self.currentAgent = -1

    def train(self, data_path_root):
        """
        """
        self.epoch += 1
        if self.epoch in self.lr_schedule.keys():
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr_schedule[self.epoch]

        trainset = RobotDatasetTrace(data_path_root)
        trainloader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        evaluateset=RobotDatasetTrace("/home/xinchi/gnn_data/evaluate")
        evaluateloader = DataLoader(
            evaluateset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.model.train()
        total_loss = 0
        total = 0
        while self.epoch < 2:
            self.epoch += 1
            iteration = 0
            for iter, batch in enumerate(tqdm(trainloader)):
                position_lists_local=batch["position_lists_local"]
                self_orientation=batch["self_orientation"]
                neighbor=batch["neighbor"]
                reference = batch["reference"]
                useless = batch["useless"]
                scale = batch["scale"]
                if self.use_cuda:
                    position_lists_local = position_lists_local.to("cuda")
                    self_orientation=self_orientation.to("cuda")
                    neighbor = neighbor.to("cuda")
                    reference = reference.to("cuda")
                    useless = useless.to("cuda")
                    scale = scale.to("cuda")
                self.optimizer.zero_grad()
                outs = self.model(position_lists_local)
                # print("input",position_lists_local)
                # print("model",outs)
                # print("expert",reference)
                loss = self.criterion(outs[0], reference[:, 0])
                # print(self.criterion(outs[0], reference[:, 0]))
                # for i in range(1, self.number_of_agent):
                #     loss += self.criterion(outs[i], reference[:, i])
                #     print(self.criterion(outs[i], reference[:, i]))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total += position_lists_local.size(0) * self.number_of_agent
                iteration += 1
                # print(iteration)
                # if iter % 1000 == 0:
                #     print("out", outs[0])
                #     print("ref",reference[:, 0])
                if iteration % 500 == 0:

                    print(
                        "Epoch "+str(self.epoch)+":" 
                        "Average training_data loss at iteration "
                        + str(iteration)
                        + ":",
                        total_loss / total,
                    )
                    print("loss ", total_loss)
                    print("total ", total)
                    total_loss=0
                    total=0
                    self.save("model_" + str(iteration) + ".pth")
                    evaluate(evaluateloader, self.use_cuda, self.model, self.optimizer, self.criterion, self.number_of_agent,
                             iteration)
        return total_loss / total

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
def evaluate(evaluateloader,use_cuda,model,optimizer,criterion,nA,iteration):
    total_loss_eval= 0
    total_eval = 0
    for iter_eval, batch_eval in enumerate(evaluateloader):
        position_lists_local = batch_eval["position_lists_local"]
        self_orientation = batch_eval["self_orientation"]
        neighbor = batch_eval["neighbor"]
        reference = batch_eval["reference"]
        useless = batch_eval["useless"]
        scale = batch_eval["scale"]
        if use_cuda:
            position_lists_local = position_lists_local.to("cuda")
            self_orientation = self_orientation.to("cuda")
            neighbor = neighbor.to("cuda")
            reference = reference.to("cuda")
            useless = useless.to("cuda")
            scale = scale.to("cuda")
        optimizer.zero_grad()
        # print(occupancy_maps.shape)

        outs = model(position_lists_local)
        loss = criterion(outs[0], reference[:, 0])

        for i in range(1, nA):
            loss += criterion(outs[i], reference[:, i])
        optimizer.step()
        total_loss_eval += loss.item()
        total_eval += position_lists_local.size(0) * nA
    print(
        "Average evaluating_data loss at iteration "
        + str(iteration)
        + ":",
        total_loss_eval / total_eval,
    )
    print("loss_eval ", total_loss_eval)
    print("total_eval ", total_eval)



if __name__ == "__main__":

    T = Trainer(load_model_path="/home/xinchi/multi_robot_formation/saved_model/model_final_expert_local_pose.pth")
    T.train(data_path_root="/home/xinchi/gnn_data/expert_adjusted_5")
    T.save("/home/xinchi/multi_robot_formation/saved_model/model_dummy.pth")
# from utils.map_viewer import visualize_global_pose_array
# trainset = RobotDatasetTrace(data_path_root="/home/xinchi/gnn_data/expert_adjusted_5")
# for i in range(99,100):
#     data,global_pose_array=trainset.__getitem__(i)
#     visualize_global_pose_array(global_pose_array)
#     print(data["reference"])
#     print(data["neighbor"])