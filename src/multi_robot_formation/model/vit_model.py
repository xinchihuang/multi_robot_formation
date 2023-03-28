import os
import sys
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation")
sys.path.append("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/model")
print(sys.path)
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm

import torch
from utils.data_generator import DataGenerator
import math
import random
import copy



from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,task="control",agent_number=5):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.task = task
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_control = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.mlp_position = nn.Sequential(
            nn.LayerNorm(dim*agent_number),
            nn.Linear(dim*agent_number, num_classes)
        )
        self.mlp_graph = nn.Sequential(
            nn.LayerNorm(agent_number),
            nn.Linear(agent_number, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if self.task=="position":
            return self.mlp_position(x)
        if self.task=="graph":
            return self.mlp_graph(x)
        if self.task=="control":
            return self.mlp_control(x)





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
        temp = copy.deepcopy(global_pose_array)
        orders = np.argsort(global_pose_array, axis=0)
        for i in range(len(orders)):
            global_pose_array[i, :] = temp[orders[i][0]]
        self_orientation_array = global_pose_array[:, 2]
        self_orientation_array = copy.deepcopy(self_orientation_array)
        global_pose_array[:, 2] = 0
        use_random = random.uniform(0, 1)
        if use_random > 0:
            global_pose_array = 2 * np.random.random((self.number_of_agents, 3)) - 1
            orders = np.argsort(global_pose_array, axis=0)
            temp = copy.deepcopy(global_pose_array)
            for i in range(len(orders)):
                global_pose_array[i, :] = temp[orders[i][0]]
            global_pose_array[:, 2] = 0
            self_orientation_array = (
                    2 * math.pi * np.random.random(self.number_of_agents) - math.pi
            )

        data_generator = DataGenerator(local=self.local, partial=self.partial)
        occupancy_maps, reference, adjacency_lists = data_generator.generate_one(
            global_pose_array, self_orientation_array
        )



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
        while self.epoch < 10:
            self.epoch += 1
            iteration = 0
            for iter, batch in enumerate(tqdm(trainloader)):
                occupancy_maps = batch["occupancy_maps"]
                reference = batch["reference"]
                if self.use_cuda:
                    occupancy_maps = occupancy_maps.to("cuda")
                    reference = reference.to("cuda")
                self.optimizer.zero_grad()
                # print(occupancy_maps.shape)
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
                    self.save("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/saved_model/"+"model_" + str(iteration)+"_epoch"+str(self.epoch) + ".pth")
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


def evaluate(evaluateloader, use_cuda, model, optimizer, criterion, number_of_agent, iteration):
    total_loss_eval = 0
    total_eval = 0
    for iter_eval, batch_eval in enumerate(evaluateloader):
        occupancy_maps = batch_eval["occupancy_maps"]
        reference = batch_eval["reference"]
        if use_cuda:
            occupancy_maps = occupancy_maps.to("cuda")
            reference = reference.to("cuda")
        optimizer.zero_grad()
        # print(occupancy_maps.shape)

        outs = model(torch.unsqueeze(occupancy_maps[:, 0, :, :], 1))
        loss = criterion(outs, reference[:, 0])

        for i in range(1, number_of_agent):
            outs = model(torch.unsqueeze(occupancy_maps[:, i, :, :], 1))
            loss += criterion(outs, reference[:, i])
        optimizer.step()
        total_loss_eval += loss.item()
        total_eval += occupancy_maps.size(0) * number_of_agent
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
        dim = 256,
        depth = 3,
        heads = 8,
        mlp_dim = 512,
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
