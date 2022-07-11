"""
A script to plot the gabreil graph changes.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


pose_array = np.load(
    ""
)
node_num = np.shape(pose_array)[0]
iters = pose_array.shape[1]
# plt.figure(figsize=(10, 10))
fig, ax = plt.subplots(figsize=(10, 10))
plt.pause(5)
for time_step in range(iters):
    gabriel_graph = [[1] * node_num for _ in range(node_num)]
    position_array = pose_array[:, time_step, :2]
    for u in range(node_num):
        for v in range(node_num):
            m = (position_array[u] + position_array[v]) / 2
            for w in range(node_num):
                if w == v:
                    continue
                if np.linalg.norm(position_array[w] - m) < np.linalg.norm(
                    position_array[u] - m
                ):
                    gabriel_graph[u][v] = 0
                    gabriel_graph[v][u] = 0
                    break
    rob_num = node_num
    ax.scatter(position_array[:, 0], position_array[:, 1])
    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt(
                (xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2
            )
            ax.plot(xlist, ylist, label=f"Distane: {distance:8 .2f}")
    ax.legend(fontsize=18)
    ax.set_title("Formation", size=20)
    ax.set_xlabel("X(m)", size=20)
    ax.set_ylabel("Y(m)", size=20)
    ax.grid()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.text(-1, 4, f"Time={time_step * 0.05:8 .2f}s", size=20)
    plt.pause(0.05)
    ax.cla()
plt.show()
