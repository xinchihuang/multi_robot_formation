
import random
from sklearn.cluster import KMeans,DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import KDTree


def create_transform_matrix(dx, dy, theta):
    rad = np.radians(theta)
    transform_matrix = np.array([
        [np.cos(rad), -np.sin(rad), dx],
        [np.sin(rad), np.cos(rad), dy],
        [0, 0, 1]
    ])
    return transform_matrix

def apply_transform(point, matrix):
    homogeneous_point = np.append(point, 1)
    transformed_point = matrix @ homogeneous_point  # Using '@' for matrix multiplication
    return transformed_point[:2]
def generate_object(number_of_point,sep1=0.02,sep2=0.25):
    point_list=[]
    point_list.append([0, sep2 / 2])
    for i in range(number_of_point):
        point_list.append([-sep1 * (number_of_point - 1) + 2*i * sep1, -sep2 / 2])
    return np.array(point_list)

def detect_objects(points,eps=0.3,sep1=0.1):
    direction_vectors_dict = defaultdict(list)
    centroids_dict=defaultdict(list)
    X = np.array(points)

    dbscan = DBSCAN(eps=eps, min_samples=3)

    # 使用DBSCAN进行聚类
    predicted_labels = dbscan.fit_predict(X)

    groups = defaultdict(list)
    centroids_dict=defaultdict(list)

    # for i in range(centroids.shape[0]):
    #     centroids_dict[i]=centroids[i]
    for i in range(len(X)):
        groups[predicted_labels[i]].append(X[i])
    direction_vectors_dict=defaultdict(list)
    for group_id in groups:
        points = np.array(groups[group_id])
        tree = KDTree(points)
        direction_vector = np.zeros((2))
        front=[]
        back=[]
        # print(points)
        for point_index in range(points.shape[0]):
            point_to_search = points[point_index]
            distance, index = tree.query(point_to_search, k=2)
            if distance[1] > 2*sep1:
                front.append(point_to_search)
                # direction_vector = direction_vector+(point_to_search - centroids_dict[group_id])
            else:
                back.append(point_to_search)
                # direction_vector = direction_vector-(point_to_search - centroids_dict[group_id])
        centroid=np.zeros((2))
        if len(front)==0 or len(back)<2:
            valid=False
            continue
        for point_back in back:
            direction_vector = direction_vector + (front[0] - point_back)
            centroid=centroid+point_back+front[0]
        centroids_dict[len(groups[group_id])]=centroid/len(back)/2
        direction_vectors_dict[len(groups[group_id])]=(direction_vector/np.linalg.norm(direction_vector))
    # print(direction_vectors_dict,centroids_dict)
    return direction_vectors_dict,centroids_dict

if __name__=="__main__":
    X=[]
    for i in range(2,6):
        trapezoid=generate_object(i)
        tr_x=random.uniform(-1,1)
        tr_y=random.uniform(-1,1)
        tr_theta=random.uniform(-360,360)
        for point in trapezoid:
            transform_matrix = create_transform_matrix(tr_x, tr_y, tr_theta)
            transformed_point = apply_transform(point, transform_matrix)
            X.append(transformed_point)
    direction_vectors_dict,X,predicted_labels,centroids_dict=detect_objects(X)
    for group_id in direction_vectors_dict:
        end1 = centroids_dict[group_id] + 0.2 * direction_vectors_dict[group_id]
        end2 = centroids_dict[group_id] - 0.2 * direction_vectors_dict[group_id]
        plt.plot([end1[0], end2[0]], [end1[1], end2[1]])
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, s=5, cmap='viridis')

    # plt.scatter(centroids[:, 0], centroids[:, 1], s=2, c='red', alpha=0.5)
    plt.xlim(-1,1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.show()