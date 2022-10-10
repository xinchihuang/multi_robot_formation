import math

import numpy as np


def global_to_local(position_lists_global):
    position_lists_local = []
    for i in range(len(position_lists_global)):
        x_self = position_lists_global[i][0]
        y_self = position_lists_global[i][1]
        z_self = position_lists_global[i][2]
        position_list_local_i = []
        for j in range(len(position_lists_global)):
            if i == j:
                continue
            position_list_local_i.append(
                [
                    position_lists_global[j][0] - x_self,
                    position_lists_global[j][1] - y_self,
                    position_lists_global[j][2] - z_self,
                ]
            )
        position_lists_local.append(position_list_local_i)
    return position_lists_local


def data_filter(world_point, max_x, max_y, max_height, min_range):

    x = world_point[0]
    y = world_point[1]
    z = world_point[2]
    if x > max_x or x < -max_x or y > max_y or y < -max_y or z < -max_height:  #
        return None
    elif x < min_range and y < min_range and x > -min_range and y > -min_range:
        return None
    return [x, y, z]


def world_to_map(world_point, map_size, max_x, max_y):

    if world_point == None:
        return None
    x_world = world_point[0]
    y_world = world_point[1]
    x_map = int((max_x - x_world) / (2 * max_x) * map_size)
    y_map = int((max_y - y_world) / (2 * max_y) * map_size)
    if 0 <= x_map < map_size and 0 <= y_map < map_size:
        return [x_map, y_map]
    return None


def generate_map(position_lists_local, robot_size, max_height, map_size, max_x, max_y):
    maps = []
    scale = min(max_x, max_y)
    robot_range = max(1, int(math.floor(map_size * robot_size / scale / 2)))
    for robot in range(len(position_lists_local)):
        occupancy_map = (
            np.ones((map_size + 2 * robot_range, map_size + 2 * robot_range)) * 255
        )
        for world_points in position_lists_local[robot]:
            world_points_filtered = data_filter(
                world_points, max_x, max_y, max_height, 2 * robot_size
            )
            map_points = world_to_map(world_points_filtered, map_size, max_x, max_y)
            if map_points == None:
                continue
            x = map_points[0]
            y = map_points[1]
            for m in range(-robot_range, robot_range, 1):
                for n in range(-robot_range, robot_range, 1):
                    occupancy_map[x + m][y + n] = 0
        occupancy_map = occupancy_map[
            robot_range:-robot_range, robot_range:-robot_range
        ]
        maps.append(occupancy_map)
    return maps


# global_positions=[[-4,-4,0],
#                 [-4,4,0],
#                 [4,4,0],
#                 [4,-4,0],
#                 [0,0,0]]
# position_lists_local=global_to_local(global_positions)
# print(position_lists_local)
# robot_size,map_size,max_x,max_y=0.2 ,100,10,10
# max_height=0.3
# maps=generate_map(position_lists_local,robot_size,max_height,map_size,max_x,max_y)
# map=maps[1]
# print(position_lists_local[1])
# for i in range(map.shape[0]):
#     line=""
#     for j in range(map.shape[1]):
#         if map[i][j]==255:
#             line+=" "
#         else:
#             line+="X"
#     print(line)
