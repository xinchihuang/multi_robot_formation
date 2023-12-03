import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from utils.gabreil_graph import get_gabreil_graph


def plot_formation_gabreil_dynamic(ax, position_array, desired_distance=1, xlim=5, ylim=5, sensor_range=2):
    rob_num = np.shape(position_array)[0]
    gabriel_graph = get_gabreil_graph(position_array, sensor_range=sensor_range)
    ax.scatter(position_array[:, 0], position_array[:, 1])
    formation_error = 0
    count = 0

    for i in range(rob_num):
        for j in range(i + 1, rob_num):
            if gabriel_graph[i][j] == 0:
                continue
            xlist = [position_array[i][0], position_array[j][0]]
            ylist = [position_array[i][1], position_array[j][1]]
            distance = math.sqrt((xlist[0] - xlist[1]) ** 2 + (ylist[0] - ylist[1]) ** 2)
            if distance > 5:
                continue
            ax.plot(xlist, ylist, label=f"Distance: {distance:.2f}")
            count += 1
            formation_error += abs(distance - desired_distance)

    ax.set_title("Average formation error: " + str(formation_error / count), fontsize=20)
    ax.set_xlabel("x(m)", fontsize=20)
    ax.set_ylabel("y(m)", fontsize=20)
    ax.legend()
    ax.set_xticks(np.arange(-xlim, xlim + 1, 1.0))
    ax.set_yticks(np.arange(-ylim, ylim + 1, 1.0))
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.grid()


trace_data=np.load("/home/xinchi/gazebo_data/ViT_1m/ViT_7_10p/50/trace.npy")
model_control_data=np.load("/home/xinchi/gazebo_data/ViT_1m/ViT_7_10p/50/model_control.npy")
print(model_control_data[0,:,:])
# def update(val):
#     time_step = int(slider.val)
#     new_position_array = position_array[time_step, :, :]  # 获取新的位置数据
#     ax.clear()  # 清除当前轴的内容
#     plot_formation_gabreil_dynamic(ax, new_position_array)  # 用新数据重新绘图
#
# # 创建图形和轴
# fig, ax = plt.subplots(figsize=(10, 10))
# plt.subplots_adjust(bottom=0.25)  # 为滑块腾出空间
#
# # 你的初始位置数组
# position_array = trace_data
# model_control_array=model_control_data
#
# # 初始绘图
# plot_formation_gabreil_dynamic(ax, position_array[0, :, :])
#
# # 创建滑块
# ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Time Step', 0, len(position_array) - 1, valinit=0, valfmt='%0.0f')
# slider.on_changed(update)
#
# plt.show()
#


