import csv
import math
import os
import sys

import numpy
import matplotlib.pyplot as plt
from scripts.utils.object_tracker import detect_objects
from plot_scene import plot_formation_gabreil_real,plot_relative_distance_gabreil_real,plot_trace_triangle_real
from collections import defaultdict
import matplotlib
for file_name in range(12,13):
    file_path = '{0}.csv'.format(file_name)
    robot_index_list=[3,4,5,6]
    # Open the CSV file and read its content
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # num_robot=6
        # Iterate over each row in the CSV file
        count=0
        object_dict=defaultdict(list)
        for i in robot_index_list:
            object_dict[i] = []
        start = False
        for row in csv_reader:
            count+=1
            if count>6000:
                break
            # remove header
            if count<=7:
                continue
            index=2
            points=[]
            # project 3d points to 2d map
            while index<len(row):
                try:
                    float(row[index])
                    float(row[index+1])
                    float(row[index+2])
                except:
                    index +=3
                    continue
                points.append([float(row[index]),-float(row[index+2])])
                index+=3
            # get detected object
            direction_vectors_dict, centroids_dict = detect_objects(points)

            if start==False:
                for item in object_dict:
                    if item in centroids_dict:
                        start=True
                    else:
                        start=False
                        break
            if start==False:
                continue
            for item in object_dict:
                if item in centroids_dict:
                    if len(object_dict[item]) == 0:
                        object_dict[item].append([centroids_dict[item][0], centroids_dict[item][1],
                                                  math.atan2(direction_vectors_dict[item][1],
                                                             direction_vectors_dict[item][1])])
                    elif ((object_dict[item][-1][0] - centroids_dict[item][0]) ** 2 + (
                            object_dict[item][-1][1] - centroids_dict[item][1]) ** 2) ** 0.5 < 0.1:
                        object_dict[item].append([centroids_dict[item][0], centroids_dict[item][1],
                                                  math.atan2(direction_vectors_dict[item][1],
                                                             direction_vectors_dict[item][1])])
                    else:
                        object_dict[item].append(object_dict[item][-1])
                else:
                    # print(item,centroids_dict)

                    if len(object_dict[item])>0:
                        object_dict[item].append(object_dict[item][-1])
                    else:
                        print(centroids_dict)
        pose_lists=[]
        for item in object_dict:
            # print(len(object_dict[item]),item)
            pose_lists.append(object_dict[item])
        if not os.path.isdir(str(file_name)):
            os.mkdir(str(file_name))

        pose_array=numpy.array(pose_lists)
        pose_array=pose_array[:,:,:]
        print(pose_array.shape)
        numpy.save(os.path.join(str(file_name), "trace"), pose_array)
        plot_trace_triangle_real(pose_array,time_step=pose_array.shape[1],xlim=2,ylim=2,save_path=str(file_name))
        plot_formation_gabreil_real(pose_array,desired_distance=1.1,xlim=2,ylim=2,save_path=str(file_name))
        plot_relative_distance_gabreil_real(0.01,pose_array,save_path=str(file_name))





