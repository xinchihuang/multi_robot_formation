import matplotlib.pyplot as plt
import numpy as np
import os
import math
import distutils.dir_util
from utils.gabreil_graph import get_gabreil_graph


def get_convergence_time_average(raw_data,desired_distance=1,tolerrance=0.1,check_timesteps=50,sensor_range=1):
    time_steps=raw_data.shape[1]
    check_window=[]
    for time_step in range(time_steps):
        data=raw_data[:,time_step,:]
        gabriel_graph = get_gabreil_graph(data,sensor_range=sensor_range)
        stop=True
        distance_error_list=[]
        for i in range(len(gabriel_graph)):
            for j in range(i, len(gabriel_graph)):
                if not i == j:
                    if gabriel_graph[i][j] == 1:
                        distance = ((data[i,0] - data[j,0])**2 + (data[i, 1] - data[j,1])**2)**0.5
                        distance_error_list.append(math.fabs(distance - desired_distance))

        if len(check_window)<check_timesteps:
            check_window.append(distance_error_list)
        else:
            sum_distance_error=0
            num_data_point=0
            for i in range(len(check_window)):
                for j in range(len(check_window[i])):
                    sum_distance_error+=check_window[i][j]
                num_data_point+=len(check_window[i])
            average_formation_error= sum_distance_error/num_data_point
            # print(time_step,sum_distance_error,num_data_point)
            if average_formation_error/desired_distance<=tolerrance:
                return time_step
            else:
                check_window.pop(0)
    return time_step


def process_data_gazebo(root_path,desired_distance=1,sensor_range=2):
    path_list=[]
    for path in os.listdir(root_path):
        path_list.append(os.path.join(root_path, path))

    converge_time_all=[]
    average_formation_all=[]
    average_formation_error_all=[]
    unsuccess=0

    for path in path_list:
        # for i in range(robot_num):
        #     if i==0:
        #         file=os.path.join(path,str(i), "trace.npy")
        #         raw_data=np.load(file)[:,0,:]
        #         raw_data=np.expand_dims(raw_data,axis=0)
        #     else:
        #         file = os.path.join(path, str(i), "trace.npy")
        #         data_i= np.load(file)[:,0,:]
        #         data_i=np.expand_dims(data_i, axis=0)
        #         raw_data=np.concatenate((raw_data,data_i),axis=0)
        raw_data=np.load(os.path.join(path, "trace.npy"))
        raw_data=raw_data.transpose((1,0,2))
        sim_time=raw_data.shape[1]*0.05
        convergence_time = get_convergence_time_average(raw_data,desired_distance=desired_distance,sensor_range=sensor_range)
        observe_data=raw_data[:,-400:,:2]
        time_steps=observe_data.shape[1]
        for time_step in range(time_steps):
            crash=False
            separate=False
            data=observe_data[:,time_step,:]
            gabriel_graph=get_gabreil_graph(data,sensor_range=sensor_range)
            reference=np.ones(data.shape[1])*desired_distance
            reference=reference
            distance_error_list=[]
            distance_list=[]
            for i in range(len(gabriel_graph)):
                for j in range(i,len(gabriel_graph)):
                    if not i==j:
                        if gabriel_graph[i][j]==1:
                            distance=np.sqrt(np.square(data[i,0]-data[j,0])+np.square(data[i,1]-data[j,1]))
                            # print(distance)
                            distance_list.append(distance)
                            if distance<0.2:
                                crash=True
                            if distance>2:
                                separate=True
                            distance_error=np.abs(distance-reference)
                            distance_error_list.append(distance_error)
        average_formation = np.average(np.array(distance_list))
        average_formation_error = 100*np.average(np.array(distance_error_list))
        # if convergence_time >= 50:
        #     unsuccess += 1
        #     print(path,average_formation_error,"no converge")
        #     to_root = "/home/xinchi/unsuccess"
        #     distutils.dir_util.copy_tree(path, os.path.join(to_root,path.split("/")[-1]))
        #     continue
        if crash==True:
            unsuccess += 1
            print(path,average_formation_error,"crash")
            # to_root = "/home/xinchi/unsuccess"
            # distutils.dir_util.copy_tree(path, os.path.join(to_root, path.split("/")[-1]))
            # continue
        if average_formation_error>10:
            unsuccess += 1
            print(path, average_formation_error,"too much error")
            # to_root = "/home/xinchi/unsuccess"
            # distutils.dir_util.copy_tree(path, os.path.join(to_root, path.split("/")[-1]))
            continue

        converge_time_all.append(convergence_time)
        average_formation_error_all.append(average_formation_error)
        average_formation_all.append(average_formation)
        # break

    print(root_path,unsuccess)

    return converge_time_all,average_formation_all,average_formation_error_all

def box_1(data_m,title,xlabel,ylabel,save_dir):
    fig = plt.figure(figsize=(5, 3))

    color_model='#1f77b4'
    color_expert='#ff7f0e'
    model=plt.boxplot(data_m,
                      positions=np.array(range(len(data_m))) * 1.0,
                      boxprops=dict(color=color_model),
                      capprops=dict(color=color_model),
                      whiskerprops=dict(color=color_model),
                      flierprops=dict(color=color_model,markeredgecolor=color_model),
                      medianprops=dict(color="black"),
                      widths=0.6)
    plt.subplots_adjust(left=0.18,
                        bottom=0.18,
                        right=0.99,
                        top=0.99,
                        wspace=0.0,
                        hspace=0.0)
    # plt.legend([model["boxes"][0], exp["boxes"][0]], ['GNN', 'Expert'], loc='upper left',borderpad=0.5,labelspacing=0.5)
    plt.xticks(np.array(range(len(data_m)))*1.0,labels=xlabel,fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title(title,fontsize=18)
    plt.xlabel("Number of robots",fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    print(save_dir)
    plt.savefig(os.path.join(save_dir,title+'.png'))
def box_2(data_m,data_e,title,ylabel,save_dir):
    fig = plt.figure(figsize=(2.5, 3))
    labels=["Expert","DNN"]
    color_model='#1f77b4'
    color_expert='#ff7f0e'
    print(title)
    print(sum(data_m[0])/len(data_m[0]))
    print(sum(data_e[0])/len(data_e[0]))
    model=plt.boxplot(data_m,
                      positions= [2.0],
                      boxprops=dict(color=color_model),
                      capprops=dict(color=color_model),
                      whiskerprops=dict(color=color_model),
                      flierprops=dict(color=color_model,markeredgecolor=color_model),
                      medianprops=dict(color="black"),
                      widths=0.4)
    exp=plt.boxplot(data_e,
                      positions= [1.0],
                      boxprops=dict(color=color_expert),
                      capprops=dict(color=color_expert),
                      whiskerprops=dict(color=color_expert),
                      flierprops=dict(color=color_expert,markeredgecolor=color_expert),
                      medianprops=dict(color="black"),
                      widths=0.4)
    # plt.legend([model["boxes"][0], exp["boxes"][0]], ['GNN', 'Expert'], loc='upper left',borderpad=2,labelspacing=2)
    plt.xticks([1,2], labels=labels, fontsize=15)
    plt.yticks(fontsize=15)
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    # plt.title(title,fontsize=18)
    # plt.xlabel("Number of robots",fontsize=18)

    plt.subplots_adjust(left=0.3,
                        bottom=0.1,
                        right=0.99,
                        top=0.99,
                        wspace=0.0,
                        hspace=0.0)
    plt.ylabel(ylabel,fontsize=15)
    plt.savefig(os.path.join(save_dir,title+'.png'))



converge_time_all_ViT=[]
average_formation_all_ViT=[]
average_formation_error_all_ViT=[]
robot_num=(7,)
root_dir="/home/xinchi/gazebo_data/ViT_1m"
for i in robot_num:
    folder="ViT_"+str(i)+"_10p"
    path = os.path.join(root_dir,folder)
    converge_time_all, average_formation_all, average_formation_error_all = process_data_gazebo(path,desired_distance=1,sensor_range=2)
    converge_time_all_ViT.append(converge_time_all)
    average_formation_all_ViT.append(average_formation_all)
    average_formation_error_all_ViT.append(average_formation_error_all)




#
box_1(converge_time_all_ViT,"Converge time",robot_num,"Convergence Time Step",root_dir)
box_1(average_formation_all_ViT,"Average distance",robot_num,"Distance(m)",root_dir)
box_1(average_formation_error_all_ViT,"Average group formation error",robot_num,"Formation Error(%)",root_dir)
# box_1(converge_time_all_expert,"Converge time","Convergence Time(s)",root_dir)
# box_1(average_formation_all_expert,"Average distance","Distance(m)",root_dir)
# box_1(average_formation_error_all_expert,"Average group formation error","Formation Error(%)",root_dir)
# box_2(converge_time_all_model,converge_time_all_expert,"Converge time 5","Convergence Time(s)",root_dir)
# box_2(average_formation_all_model,average_formation_all_expert,"Average distance 5","Distance(m)",root_dir)
# box_2(average_formation_error_all_model,average_formation_error_all_expert,"Average group formation error 5","Formation Error(%)",root_dir)