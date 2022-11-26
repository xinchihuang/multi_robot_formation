import torch
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from model.GNN_based_model import DecentralController

class RobotDataset(Dataset):

    def __init__(self, data1, data2, data3, data4, data5, nA, inW = 100, inH = 100, transform = None):
        self.obs = data1
        self.inW = inW
        self.inH = inH
        c1 = np.zeros((self.obs.shape[0]//nA,nA,self.inW,self.inH))
        jump = c1.shape[0]
        print('Arranging Data')
        for i in tqdm(range(len(c1))):
            for j in range(nA):
                c1[i,j] = self.obs[(j * jump) + i].reshape((self.inW, self.inH))
        self.obs = c1.copy()

        self.gt = data2
        c2 = np.zeros((self.obs.shape[0],nA,2))
        for i in tqdm(range(len(c2))):
            for j in range(nA):
                c2[i,j] = self.gt[(j * jump) + i]
        self.gt = c2.copy()
        self.graphs = data3
        c3 = np.zeros((self.obs.shape[0],nA, nA,nA))
        for i in tqdm(range(len(c3))):
            for j in range(nA):
                c3[i,j] = self.graphs[(j * jump) + i].reshape((nA,nA))
        self.graphs = c3.copy()

        self.refs = data4
        c4 = np.zeros((self.obs.shape[0],nA,1))
        for i in tqdm(range(len(c4))):
            for j in range(nA):
                c4[i,j,0] = self.refs[(j * jump) + i]
        self.refs = c4.copy()

        self.alphas = data5
        c5 = np.zeros((self.obs.shape[0],nA,1))
        for i in tqdm(range(len(c5))):
            for j in range(nA):
                c5[i,j,0] = self.alphas[(j * jump) + i]
        self.alphas = c5.copy()
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):

        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        sample = self.obs[idx]
        gt = self.gt[idx]
        graph = self.graphs[idx]
        refs = self.refs[idx]
        alphas = self.alphas[idx]

        if(self.transform):
            #for i in range(3):
                #s = sample[:,i]
                #m = np.mean(s)
                #std = np.std(s)
                #sample[:,i] = (s - m)/(std + .00001)
            sample = torch.from_numpy(sample).double()
            gt = torch.from_numpy(gt).double()
            graph = torch.from_numpy(graph).double()
            refs = torch.from_numpy(refs).double()
            alphas = torch.from_numpy(alphas).double()
        return {'data':sample, 'graphs':graph,'actions':gt, 'refs':refs, 'alphas':alphas}


class Trainer:
    def __init__(
        self,
        criterion="mse",
        optimizer="rms",
        inW=100,
        inH=100,
        batch_size=16,
        nA=3,
        lr=0.01,
        cuda=True,
    ):
        self.points_per_ep = None
        self.nA = nA
        self.inW = inW
        self.inH = inH
        self.batch_size = batch_size
        self.model = DecentralController(number_of_agent=self.nA, input_width=self.inW, input_height=self.inH,use_cuda=False).double()
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


    def train(self, data):
        """
        datalist[0].d['actions', 'graph', 'observations']
        """
        self.epoch += 1
        if self.epoch in self.lr_schedule.keys():
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr_schedule[self.epoch]
        actions = data[0].d["actions"]
        inputs = data[0].d["observations"]
        graphs = data[0].d["graph"]
        refs = data[0].d["obs2"][:, 1]
        alphas = data[0].d["obs2"][:, 2]
        # np.save('actions.npy', actions)
        # np.save('inputs.npy', inputs)
        # np.save('graphs.npy', graphs)
        trainset = RobotDataset(
            inputs,
            actions,
            graphs,
            refs,
            alphas,
            self.nA,
            inW=self.inW,
            inH=self.inH,
            transform=self.transform,
        )
        trainloader = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.model.train()
        total_loss = 0
        total = 0
        print("training")
        iteration = 0
        for i, batch in enumerate(tqdm(trainloader)):
            iteration += 1
            inputs = batch["data"].to("cuda")
            S = batch["graphs"][:, 0, :, :].to("cuda")
            actions = batch["actions"].to("cuda")
            refs = batch["refs"].to("cuda")
            alphas = batch["alphas"].to("cuda")
            self.model.addGSO(S)
            self.optimizer.zero_grad()
            outs = self.model(inputs, refs, alphas)
            print(outs[0], actions[:, 0])
            loss = self.criterion(outs[0], actions[:, 0])
            for i in range(1, self.nA):
                loss += self.criterion(outs[i], actions[:, i])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            total += inputs.size(0) * self.nA
        print(iteration)
        print("Average training loss:", total_loss / total)
        return total_loss / total

    def save(self, pth):
        torch.save(self.model.state_dict(), pth)


Trainer.train(data)