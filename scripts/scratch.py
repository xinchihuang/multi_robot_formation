# 导入必要的库
import math
import random

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 创建一个数据集
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)



point = np.array([2, 3])

def create_transform_matrix(dx, dy, theta):
    # 角度转换为弧度
    rad = np.radians(theta)
    # 创建仿射变换矩阵
    transform_matrix = np.array([
        [np.cos(rad), -np.sin(rad), dx],
        [np.sin(rad), np.cos(rad), dy],
        [0, 0, 1]
    ])
    return transform_matrix

def apply_transform(point, matrix):
    # 将点转换为齐次坐标
    homogeneous_point = np.append(point, 1)
    # 应用变换矩阵
    transformed_point = matrix @ homogeneous_point  # Using '@' for matrix multiplication
    return transformed_point[:2]
def generate_trapezoid(number_of_point,sep1=0.1,sep2=1):
    point_list=[]
    for i in range(number_of_point):
        point_list.append([-sep1*(number_of_point-1)/2+i*sep1,sep2/2])
        point_list.append([-sep1 * (number_of_point - 1) + 2*i * sep1, -sep2 / 2])
    return np.array(point_list)
print(generate_trapezoid(2))
X=[]
y=[]
for i in range(2,6):
    trapezoid=generate_trapezoid(i)
    tr_x=random.uniform(-3,3)
    tr_y=random.uniform(-3,3)
    tr_theta=random.uniform(-360,360)
    for point in trapezoid:
        transform_matrix = create_transform_matrix(tr_x, tr_y, tr_theta)
        # 应用变换
        transformed_point = apply_transform(point, transform_matrix)
        X.append(transformed_point)
        y.append(i)
X=np.array(X)
y=np.array(y)
kmeans = KMeans(n_clusters=4)

# 拟合模型
kmeans.fit(X)

# 预测簇标签
predicted_labels = kmeans.predict(X)

# 获取簇的中心点
centroids = kmeans.cluster_centers_
# print(centroids)
plt.xlim(-5,5)
plt.ylim(-5,5)
# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', alpha=0.5)
plt.show()