import math
import time
import numpy as np

from object_tracker import detect_objects
from collections import defaultdict
from gabreil_graph import get_gabreil_graph_dict
def centralized_control(mocap_data,desired_distance=1):
    #### detect robot pose
    point_list = []
    for marker_id in range(len(mocap_data.labeled_marker_data.labeled_marker_list)):
        # print(mocap_data.labeled_marker_data.labeled_marker_list[marker_id].pos)
        point_list.append([mocap_data.labeled_marker_data.labeled_marker_list[marker_id].pos[0],
                           mocap_data.labeled_marker_data.labeled_marker_list[marker_id].pos[2]])
    # print(point_list)
    direction_vectors_dict, centroids_dict = detect_objects(point_list)

    #### broadcast robot pose
    message_to_broadcast = ''
    object_dict=defaultdict(list)
    for object_id in centroids_dict:
        position_list = centroids_dict[object_id]
        direction_vector = direction_vectors_dict[object_id]
        orientation = -math.atan2(direction_vector[1], direction_vector[0])
        # print(object_id,math.degrees(orientation))
        # time.sleep(1)
        pose = [centroids_dict[object_id][0], -centroids_dict[object_id][1], orientation]
        object_dict[object_id]=pose
        # message_to_broadcast += str(object_id) + ":" + str(pose) + ";"
    gabreil_dict=get_gabreil_graph_dict(object_dict)
    for object_id in object_dict:
        control_global = np.zeros(2)
        object_self=np.array(object_dict[object_id])[:2]
        # print(gabreil_dict)
        for neighbor_id in gabreil_dict[object_id]:

            neighbor=object_dict[neighbor_id][:2]
            # print(object_dict)
            distance_formation = np.linalg.norm(neighbor-object_self)
            rate_f = (distance_formation - desired_distance) / distance_formation
            control_global=control_global+rate_f*(neighbor-object_self)
        #     print(control_global)
        # print(control_global,"c")
        if all(control_global)==0:
            transformed_control=control_global
        else:
            transformed_control=control_global@np.array([[math.cos(np.array(object_dict[object_id])[2]),-math.sin(np.array(object_dict[object_id])[2])],
                                                      [math.sin(np.array(object_dict[object_id])[2]),math.cos(np.array(object_dict[object_id])[2])]])

        message_to_broadcast += str(object_id) + ":" + str(list(transformed_control)) + ";"
    # print(message_to_broadcast)
    # time.sleep(0.1)
    return message_to_broadcast