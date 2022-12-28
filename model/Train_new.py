import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from model.GNN_based_model import DecentralController
import os
from utils.data_generator import generate_one

class RobotDatasetTrace(Dataset):

    def __init__(self, data_path_root):

        self.transform=True
        self.desired_distance=2
        self.number_of_agents=5

        self.num_sample = len(os.listdir(data_path_root))
        self.occupancy_maps_list=[]
        self.pose_array=np.empty(shape=(5,1,3))
        self.reference_control_list=[]
        self.neighbor_list=[]
        self.scale = np.zeros((self.number_of_agents, 1))
        for i in range(self.number_of_agents):
            self.scale[i, 0] = self.desired_distance
        for sample_index in tqdm(range(self.num_sample)):
            data_sample_path = os.path.join(data_path_root, str(sample_index))
            pose_array_i=np.load(os.path.join(data_sample_path,"pose_array_scene.npy"))
            if self.pose_array.shape[1]==1:
                self.pose_array=pose_array_i
                continue
            self.pose_array=np.concatenate((self.pose_array,pose_array_i),axis=1)
        print(self.pose_array.shape)
    def __len__(self):
        return self.pose_array.shape[1]

    def __getitem__(self,idx):

        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        global_pose_array=self.pose_array[:,idx,:]
        self_orientation_array = global_pose_array[:,2]
        global_pose_array[:,2]=0
        occupancy_maps,reference,adjacency_lists=generate_one(global_pose_array, self_orientation_array, self.desired_distance)

        neighbor = np.zeros((self.number_of_agents, self.number_of_agents))
        for key, value in adjacency_lists[0].items():
            for n in value:
                neighbor[key][n[0]] = 1
        refs = np.zeros((self.number_of_agents, 1))


        alphas = self.scale

        if(self.transform):
            #for i in range(3):
                #s = sample[:,i]
                #m = np.mean(s)
                #std = np.std(s)
                #sample[:,i] = (s - m)/(std + .00001)
            occupancy_maps= torch.from_numpy(occupancy_maps).double()
            reference = torch.from_numpy(reference).double()
            neighbor = torch.from_numpy(neighbor).double()
            refs = torch.from_numpy(refs).double()
            alphas = torch.from_numpy(alphas).double()
        return {'occupancy_maps':occupancy_maps, 'neighbor': neighbor,'reference':reference, 'useless':refs, 'scale':alphas}


class RobotDatasetMap(Dataset):

    def __init__(self, data_path_root):

        self.transform=True
        self.desired_distance=1
        self.number_of_agents=5

        self.num_sample = len(os.listdir(data_path_root))
        self.occupancy_maps_list=[]
        self.pose_list=[]
        self.reference_control_list=[]
        self.neighbor_list=[]
        self.scale = np.zeros((self.number_of_agents, 1))
        for i in range(self.number_of_agents):
            self.scale [i, 0] = self.desired_distance
        for sample_index in tqdm(range(self.num_sample)):
            data_sample_path = os.path.join(data_path_root, str(sample_index))
            occupancy_maps_i=np.load(os.path.join(data_sample_path,"occupancy_maps.npy"))
            adjacency_lists_i = np.load(os.path.join(data_sample_path, "adjacency_lists.npy"),allow_pickle=True)
            reference_controls_i = np.load(os.path.join(data_sample_path, "reference_controls.npy"))
            # self.occupancy_maps_list.append(np.expand_dims(occupancy_maps_i,axis=0))
            self.occupancy_maps_list.append(occupancy_maps_i)
            self.reference_control_list.append(reference_controls_i)
            neighbor = np.zeros((self.number_of_agents, self.number_of_agents))
            for key, value in adjacency_lists_i[0].items():
                for n in value:
                    neighbor[key][n[0]] = 1

            self.neighbor_list.append(neighbor)
    def __len__(self):
        return self.num_sample

    def __getitem__(self,idx):

        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        print(idx)
        occupancy_maps = self.occupancy_maps_list[idx]
        reference = self.reference_control_list[idx]
        neighbor = self.neighbor_list[idx]
        refs = np.zeros((self.number_of_agents, 1))
        alphas = self.scale
        if(self.transform):
            #for i in range(3):
                #s = sample[:,i]
                #m = np.mean(s)
                #std = np.std(s)
                #sample[:,i] = (s - m)/(std + .00001)
            occupancy_maps= torch.from_numpy(occupancy_maps).double()
            reference = torch.from_numpy(reference).double()
            neighbor = torch.from_numpy(neighbor).double()
            refs = torch.from_numpy(refs).double()
            alphas = torch.from_numpy(alphas).double()
        return {'occupancy_maps':occupancy_maps, 'neighbor': neighbor,'reference':reference, 'useless':refs, 'scale':alphas}
class Trainer:
    def __init__(
        self,
        criterion="mse",
        optimizer="rms",
        inW=100,
        inH=100,
        batch_size=16,
        nA=5,
        lr=0.01,
        cuda=True,
    ):
        self.points_per_ep = None
        self.nA = nA
        self.inW = inW
        self.inH = inH
        self.batch_size = batch_size
        self.model = DecentralController(number_of_agent=self.nA, input_width=self.inW, input_height=self.inH,use_cuda=cuda).double()
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
        datalist[0].d['actions', 'graph', 'observations']
        """
        self.epoch += 1
        if self.epoch in self.lr_schedule.keys():
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr_schedule[self.epoch]

        trainset = RobotDatasetTrace(data_path_root)
        trainloader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.model.train()
        total_loss = 0
        total = 0
        while self.epoch <10:
            self.epoch+=1
            iteration=0
            for i, batch in enumerate(tqdm(trainloader)):
                occupancy_maps = batch["occupancy_maps"]
                neighbor = batch["neighbor"]
                reference = batch["reference"]
                useless = batch["useless"]
                scale = batch["scale"]

                if self.use_cuda:

                    occupancy_maps=occupancy_maps.to("cuda")
                    neighbor=neighbor.to("cuda")
                    reference=reference.to("cuda")
                    useless=useless.to("cuda")
                    scale=scale.to("cuda")
                self.model.addGSO(neighbor)
                self.optimizer.zero_grad()
                outs = self.model(occupancy_maps, useless, scale)
                # print(outs[0], actions[:, 0])
                loss = self.criterion(outs[0], reference[:,0])
                for i in range(1, self.nA):
                    loss += self.criterion(outs[i], reference[:,i])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total += occupancy_maps.size(0) * self.nA
                iteration+=1
            # print(iteration)
                if iteration%100==0:
                    print("Average training_data loss at iteration "+str(iteration)+":", total_loss / total)
                    self.save("model_"+str(iteration)+".pth")
        return total_loss / total

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
T=Trainer()
T.train(data_path_root="/home/xinchi/gnn_data/expert_adjusted_5")
T.save("model_final.pth")