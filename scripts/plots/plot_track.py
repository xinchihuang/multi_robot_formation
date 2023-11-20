import csv
import math
import sys

import numpy
import matplotlib.pyplot as plt
from scripts.utils.object_tracker import detect_objects
from plot_scene import plot_formation_gabreil,plot_relative_distance_gabreil,plot_trace_triangle
from collections import defaultdict
import matplotlib
# Replace 'your_file.csv' with the path to your CSV file
file_path = 'Square.csv'
robot_index_list=[6]
# Open the CSV file and read its content
with (open(file_path, 'r') as file):
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    num_robot=6
    # Iterate over each row in the CSV file
    count=0
    object_dict=defaultdict(list)
    for i in robot_index_list:
        object_dict[i] = []
    for row in csv_reader:
        count+=1
        if count<=7:
            continue
        index=2
        points=[]
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
        direction_vectors_dict, centroids_dict = detect_objects(points)
        # print(len(points))
        # print(points)
        # print(direction_vectors_dict)
        # print(centroids_dict)

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
                print(item)
                object_dict[item].append(object_dict[item][-1])
    pose_lists=[]
    for item in object_dict:
        print(len(object_dict[item]),item)
        pose_lists.append(object_dict[item])
    pose_array=numpy.array(pose_lists)
    plot_trace_triangle(pose_array,xlim=2,ylim=2)
    plot_formation_gabreil(pose_array,desired_distance=1,xlim=2,ylim=2)





