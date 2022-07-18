import torch
import numpy as np
from graphs.models import suhaas_model
from torch.utils.data import DataLoader
from torchvision import transforms
import custom_dataset
import torch.nn as nn
from tqdm import tqdm


class Agent:
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
        self.model = suhaas_model.DecentralPlannerNet(
            nA=self.nA, inW=self.inW, inH=self.inH
        ).double()
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

    ### model controller
    def test(self, omlist, index):
        self.currentAgent += 1
        self.currentAgent = self.currentAgent % self.nA
        x = np.zeros((1, self.nA, self.inW, self.inH))
        S = omlist[index][1]
        r = np.zeros((1, self.nA, 1))
        a = np.zeros((1, self.nA, 1))
        for i in range(self.nA):
            x[0, i, :, :] = omlist[i][0].reshape((self.inW, self.inH))
            r[0, i, 0] = omlist[i][2]
            a[0, i, 0] = omlist[i][3]
        xin = torch.from_numpy(x).double()
        if self.use_cuda:
            xin = xin.to("cuda")
        S = np.array(S)
        S = S.reshape((self.nA, self.nA))
        S = torch.from_numpy(S)
        S = S.unsqueeze(0)
        if self.use_cuda:
            S = S.to("cuda")

        r = torch.from_numpy(r).double()
        if self.use_cuda:
            r = r.to("cuda")

        a = torch.from_numpy(a).double()
        if self.use_cuda:
            a = a.to("cuda")
        self.model.eval()
        self.model.addGSO(S)

        #### Set a threshold to eliminate small movements
        # threshold=0.05
        control = self.model(xin, r, a)[index]  ## model output
        # torch.where(control<threshold, 0., control)
        # torch.where(control>-threshold, 0., control)
        outs = [control]
        # print("Control",outs)
        return outs

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
        trainset = custom_dataset.RobotDataset(
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
