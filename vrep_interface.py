"""
A interface to operate Vrep APIs
author: Xinchi Huang
"""
import time

import sim as vrep

object_names = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor','velodyneVPL_16']
def init_vrep():
    print ('Program started')
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_ID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    if client_ID!=-1:
        print('Connected to remote API server')
        vrep.simxSynchronous(client_ID, True)
        vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)
    return client_ID
def get_vrep_handle(client_ID, robot_index):
    handle_name_suffix='#' + str(robot_index-1)
    if robot_index==0:
        handle_name_suffix=""
    res1, robot_handle = vrep.simxGetObjectHandle(
        client_ID, 'Pioneer_p3dx' + handle_name_suffix,
        vrep.simx_opmode_oneshot_wait)
    res2, motor_left_handle = vrep.simxGetObjectHandle(
        client_ID, 'Pioneer_p3dx_leftMotor' + handle_name_suffix,
        vrep.simx_opmode_oneshot_wait)
    res3, motor_right_handle = vrep.simxGetObjectHandle(
        client_ID, 'Pioneer_p3dx_rightMotor' + handle_name_suffix,
        vrep.simx_opmode_oneshot_wait)
    res4, point_cloud_handle = vrep.simxGetObjectHandle(
        client_ID, 'velodyneVPL_16' + handle_name_suffix,
        vrep.simx_opmode_oneshot_wait)
    print("Vrep res: ", res1, res2, res3,res4)
    return robot_handle,motor_left_handle, motor_right_handle,point_cloud_handle
def get_robot_pose(client_ID,robot_handle):
    res, pos = vrep.simxGetObjectPosition(client_ID,robot_handle, -1,vrep.simx_opmode_blocking)
    res, ori = vrep.simxGetObjectOrientation(client_ID,robot_handle, -1,vrep.simx_opmode_blocking)
    return pos,ori
def get_sensor_data(client_ID,robot_handle,robot_index):
    handle_name_suffix = '#' + str(robot_index-1)
    if robot_index==0:
        handle_name_suffix=""
    res, vel, omega = vrep.simxGetObjectVelocity(client_ID,robot_handle,vrep.simx_opmode_blocking)
    velodyne_points = vrep.simxCallScriptFunction(client_ID, 'velodyneVPL_16' + handle_name_suffix, 1,'getVelodyneData_function',
                                                  [], [], [], 'abc',vrep.simx_opmode_blocking)
    return vel, omega, velodyne_points

def post_robot_setting():

    return
def post_robot_pose(client_ID,robot_handle,position,orientation):

    vrep.simxSetObjectPosition(client_ID, robot_handle, -1,
                               position, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectOrientation(client_ID, robot_handle, -1,
                                  orientation, vrep.simx_opmode_oneshot)

    return
def post_control(client_ID,motor_left_handle,motor_right_handle, omega1, omega2):
    vrep.simxSetJointTargetVelocity(client_ID,
                                    motor_left_handle,
                                    omega1, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(client_ID,
                                    motor_right_handle,
                                    omega2, vrep.simx_opmode_oneshot)
    return
def synchronize(clinet_ID):
    vrep.simxSynchronousTrigger(clinet_ID)

# client_ID=init_vrep()
# robot_index=1
# robot_handle, motor_left_handle, motor_right_handle, point_cloud_handle = get_vrep_handle(client_ID, robot_index)
#
# while True:
#
#     pos,ori=get_robot_pose(client_ID,robot_handle)
#     vel, omega, velodyne_points=get_sensor_data(client_ID,robot_handle,robot_index)
#     print(pos, ori)
#     post_control(client_ID, motor_left_handle, motor_right_handle, 10, 10)
#     synchronize(client_ID)

