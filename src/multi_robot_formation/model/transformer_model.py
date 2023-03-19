import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
print(sys.path)


import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_robot_formation.model.nalu import NALU
from multi_robot_formation.model.weights_initializer import weights_init



class DummyModel(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        print("using dummy model controller")
        self.use_cuda = use_cuda
        self.device = "cuda" if use_cuda else "cpu"
        self.NALU_layers=NALU(3,2,100,2)
    def forward(self, input_tensor,d):
        # rate=(1-d/torch.norm(input_tensor,dim=1))
        # print(input_tensor)
        action_current=self.NALU_layers(input_tensor)
        # print(action_current)
        # action_current=rate*action_current
        return action_current

def expert(input,d):
    # print(input)
    rate=(1-d/torch.norm(input,dim=1))
    # print(rate)
    # print(torch.unsqueeze(torch.sum(input*torch.unsqueeze(rate,-1),0),0))
    return torch.unsqueeze(torch.sum(input*torch.unsqueeze(rate,-1),0),0)



class Trainer:
    def __init__(
        self,
        model,
    ):
        self.use_cuda=True
        self.model = model
        self.learning_rate = 0.000001
        self.max_iteration=100000
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate)
    def train(self):
        """ """
        iteration = 0
        while iteration<self.max_iteration:
            if self.use_cuda:
                self.model.to("cuda")
            # self.optimizer.zero_grad()
            input = torch.rand(1,2)*10 - 5
            d = torch.tensor([2])
            reference=expert(input,d)
            if self.use_cuda:
                d=d.to("cuda")
                input=input.to("cuda")
                reference=reference.to("cuda")
            outs = self.model(input)
            # print(outs,reference)
            loss = self.criterion(outs, reference)
            loss.backward()
            if iteration%1000==0:
                total_loss = 0
                for _ in range(100):
                    input = torch.rand(1, 2) * 10 - 5
                    d = torch.tensor([2])
                    reference = expert(input,d)
                    if self.use_cuda:
                        d = d.to("cuda")
                        input = input.to("cuda")
                        reference = reference.to("cuda")
                    outs = self.model(input)
                    # print(input,outs,reference)
                    total_loss += self.criterion(outs, reference).item()
                print("average loss ",total_loss/100)
            self.optimizer.step()
            iteration += 1


    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

model=NALU(3,2,10,2)
#
#
trainer=Trainer(model)
trainer.train()
#
# input = torch.ones(4, 2)-1.0000001
# d = torch.tensor([2])
# reference = expert(input,d)