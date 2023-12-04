import random
import math
import os

import numpy as np
import multiprocessing
import signal
from utils.gabreil_graph import get_gabreil_graph_local,get_gabreil_graph
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


def check_valid_initial_graph(graph_local):
    valid=True
    connected_component=find_weakly_connected_components(graph_local)
    if len(connected_component)>1:
        valid=False
    return valid
def initialize_pose(num_robot, initial_max_range=2,initial_min_range=1,sensor_range=2):
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

        gabriel_graph_global = get_gabreil_graph(pose_list,sensor_range=sensor_range)
        # gabriel_graph_local = get_gabreil_graph_local(pose_list)

        if check_valid_initial_graph(gabriel_graph_global)==True:
            for line in gabriel_graph_global:
                print(line)
            # print("----------")
            # for line in gabriel_graph_local:
            #     print(line)
            break
    print(pose_list)
    return pose_list
def initialize_pose_multi(queue,num_robot, initial_max_range=2,initial_min_range=1,sensor_range=2):
    ignore_sigint()
    while True:
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

            gabriel_graph_global = get_gabreil_graph(pose_list,sensor_range=sensor_range)
            # gabriel_graph_local=get_gabreil_graph_local(pose_list)

            if check_valid_initial_graph(gabriel_graph_global)==True:
                for line in gabriel_graph_global:
                    print(line)
                # # print("----------")
                # for line in gabriel_graph_local:
                #     print(line)
                break
        # print(multiprocessing.current_process().name)
        queue.put(pose_list)
def initialize_pose_pentagon(queue,radius=2):
    ignore_sigint()
    while True:

        pose_list=[[0,0,2 * math.pi * random.uniform(0, 1) - math.pi],
                   [2*radius * np.cos(2 * math.pi * random.uniform(0, 1)),2*radius * np.sin(2 * math.pi * random.uniform(0, 1)),2 * math.pi * random.uniform(0, 1) - math.pi]]
        num_vertices = 5
        phase=random.random()*2 * np.pi
        for i in range(num_vertices):
            theta = 2 * math.pi * random.uniform(0, 1) - math.pi
            angle = (2 * np.pi * i / num_vertices)+phase
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pose_list.append([x,y,theta])
        # print(multiprocessing.current_process().name)
        queue.put(pose_list)
        # print(pose_list)
def initial_from_data(root):
    for i in range(len(os.listdir(root))):
        if i==0:
            pose_array_data=np.load(os.path.join(root,os.listdir(root)[i]))
        else:
            pose_array_data=np.concatenate((pose_array_data,np.load(os.path.join(root,os.listdir(root)[i]))))
    print(pose_array_data.shape)
    return pose_array_data
def generate_valid_pose(root,num_robot=5,initial_max_range=5.,initial_min_range=1.):
    if not os.path.exists(root):
        os.mkdir(root)
    count = len(os.listdir(root))*100
    pose_list_to_save=[]
    while True:
        pose_list=initialize_pose(num_robot,initial_max_range=initial_max_range,initial_min_range=initial_min_range)
        pose_list_to_save.append(pose_list)
        count+=1
        print(count)
        if count%100==0:
            pose_file = os.path.join(root, str(count + 100))
            pose_array=np.array(pose_list_to_save)
            np.save(pose_file,pose_array)
            pose_list_to_save=[]
def valid_pose_saver(queue,root):
    pose_list_to_save = []
    while True:
        pose_list=queue.get()
        pose_list_to_save.append(pose_list)
        if len(pose_list_to_save) == 100:
            pose_file = os.path.join(root, str((len(os.listdir(root))+1)*100))
            print((len(os.listdir(root))+1)*100)
            pose_array = np.array(pose_list_to_save)
            np.save(pose_file, pose_array)
            pose_list_to_save = []

def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class PoseDataLoader:
    def __init__(self,root_list):
        first=True
        for root in root_list:
            for i in range(len(os.listdir(root))):
                if first:
                    pose_array_data = np.load(os.path.join(root, os.listdir(root)[i]))
                    first=False
                else:
                    pose_array_data = np.concatenate((pose_array_data, np.load(os.path.join(root, os.listdir(root)[i]))))
        self.data=pose_array_data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
class TraceDataLoader:
    def __init__(self,root_list):
        first=True
        for root in root_list:
            for i in range(len(os.listdir(root))):
                if first:
                    pose_array_data = np.load(os.path.join(root, os.listdir(root)[i],"trace.npy"))
                    first=False
                else:
                    pose_array_data = np.concatenate((pose_array_data, np.load(os.path.join(root, os.listdir(root)[i],"trace.npy"))))
        self.data=pose_array_data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

# def init_worker():
#     signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == "__main__":
    root="poses"
    if not os.path.exists(root):
        os.mkdir(root)
    num_robot = 7
    initial_max_range = 2
    initial_min_range = 1
    sensor_range=2
    # initialize_pose(num_robot,  initial_max_range=initial_max_range,  initial_min_range=initial_min_range, sensor_range=sensor_range)
    #
    queue = multiprocessing.Queue()
    num_process = 4  # Number of random number generating processes

    # Start the writer process
    writer_process = multiprocessing.Process(target=valid_pose_saver, args=(queue, root))
    writer_process.start()

    processes = []
    for _ in range(num_process):  # Four generator processes
        p = multiprocessing.Process(target=initialize_pose_multi, args=(queue,num_robot,initial_max_range,initial_min_range))
        processes.append(p)
        p.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, terminating processes...")

        # Terminate generator processes
        for p in processes:
            p.terminate()

        # Notify the writer process to terminate
        queue.put("DONE")
        writer_process.join()

    # print("All processes completed.")
    # initialize_pose_pentagon(queue, radius=2)
    # initialize_pose(5)
    # generate_valid_pose("poses_large_7",num_robot=7,initial_max_range=5,initial_min_range=0.5)
    # initial_from_data("poses")