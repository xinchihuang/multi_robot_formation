import random
import math
import os

import numpy as np

from ..utils.gabreil_graph import get_gabreil_graph_local,get_gabreil_graph
# from gabreil_graph import get_gabreil_graph_local,get_gabreil_graph
def dfs(node, visited, adjacency_matrix, component):
    visited[node] = True
    component.add(node)
    for neighbor, connected in enumerate(adjacency_matrix[node]):
        if connected and not visited[neighbor]:
            dfs(neighbor, visited, adjacency_matrix, component)

def find_weakly_connected_components(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    visited = [False] * num_nodes
    components = []

    for node in range(num_nodes):
        if not visited[node]:
            component = set()
            dfs(node, visited, adjacency_matrix, component)
            components.append(component)

    return components
def is_graph_balanced(adjacency_matrix):
    num_nodes = len(adjacency_matrix)

    for node in range(num_nodes):
        indegree = sum(adjacency_matrix[i][node] for i in range(num_nodes))
        outdegree = sum(adjacency_matrix[node][i] for i in range(num_nodes))

        if indegree != outdegree:
            return False

    return True
def is_gabriel(graph_global,graph_local):
    for i in range(len(graph_global)):
        for j in range(i+1,len(graph_global)):
            # print(i, j,graph_global[i][j])
            if graph_global[i][j]==1:
                if graph_local[i][j]==0 and graph_local[j][i]==0:
                    return False
    return True


def check_valid_initial_graph(graph_global,graph_local):
    valid=True
    connected_component=find_weakly_connected_components(graph_local)
    if is_graph_balanced(graph_local) == False:
        valid = False
    if len(connected_component)>1:
        valid=False
    if is_gabriel(graph_global,graph_local)==False:
        valid=False
    return valid
def initialize_pose(num_robot, initial_max_range=5,initial_min_range=1):

    while True:
        pose_list = []
        for i in range(num_robot):
            while True:
                redo = False
                x = 2 * random.uniform(0, 1) * initial_max_range - initial_max_range
                y = 2 * random.uniform(0, 1) * initial_max_range - initial_max_range
                theta = 2 * math.pi * random.uniform(0, 1) - math.pi
                min_distance = float("inf")
                if i == 0:
                    pose_list.append([x, y, theta])
                    break
                for j in range(len(pose_list)):
                    distance = ((x - pose_list[j][0]) ** 2 + (y - pose_list[j][1]) ** 2) ** 0.5
                    if distance < initial_min_range:
                        redo = True
                        break
                    if min_distance > distance:
                        min_distance = distance
                if redo==False:
                    pose_list.append([x,y,theta])
                    break

        gabriel_graph_global = get_gabreil_graph(pose_list)
        gabriel_graph_local=get_gabreil_graph_local(pose_list)

        if check_valid_initial_graph(gabriel_graph_global,gabriel_graph_local)==True:
            for line in gabriel_graph_global:
                print(line)
            print("----------")
            for line in gabriel_graph_local:
                print(line)
            break
    return pose_list
def initial_from_data(root):
    print(os.listdir(root))
    for i in range(len(os.listdir(root))):
        if i==0:
            pose_array_data=np.load(os.path.join(root,os.listdir(root)[i]))
        else:
            print()
            pose_array_data=np.concatenate((pose_array_data,np.load(os.path.join(root,os.listdir(root)[i]))))
    print(pose_array_data.shape)
def generate_valid_pose(root,num_robot=7):
    if not os.path.exists(root):
        os.mkdir(root)
    count = len(os.listdir(root))*100
    pose_list_to_save=[]
    while True:
        pose_list=initialize_pose(num_robot)
        pose_list_to_save.append(pose_list)
        count+=1
        print(count)
        if count%100==0:
            pose_file = os.path.join(root, str(count + 100))
            pose_array=np.array(pose_list_to_save)
            np.save(pose_file,pose_array)
            pose_list_to_save=[]
class PoseDataLoader:
    def __init__(self,root):
        for i in range(len(os.listdir(root))):
            if i == 0:
                pose_array_data = np.load(os.path.join(root, os.listdir(root)[i]))
            else:
                pose_array_data = np.concatenate((pose_array_data, np.load(os.path.join(root, os.listdir(root)[i]))))
        self.data=pose_array_data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    # initialize_pose(5)
    generate_valid_pose("poses")
    # initial_from_data("poses")