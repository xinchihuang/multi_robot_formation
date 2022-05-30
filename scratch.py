"""
author: Xinchi Huang
"""
import sim as vrep
objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
def initVrep():
    print ('Program started')
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    if clientID!=-1:
        print('Connected to remote API server')
        vrep.simxSynchronous(clientID, True)
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    return clientID
#### Get Vrep handles from simulator and pass them to robot_old.py. Handle group of parameters in simulator to define robot and sensor.
def setVrepHandles(objectNames,clientID, handleNameSuffix = ""):
    handleNames = objectNames
    res1, robotHandle = vrep.simxGetObjectHandle(
            clientID, handleNames[0] + handleNameSuffix,
            vrep.simx_opmode_oneshot_wait)

    res2, motorLeftHandle = vrep.simxGetObjectHandle(
            clientID, handleNames[1] + handleNameSuffix,
            vrep.simx_opmode_oneshot_wait)
    res3, motorRightHandle = vrep.simxGetObjectHandle(
            clientID, handleNames[2] + handleNameSuffix,
            vrep.simx_opmode_oneshot_wait)
    print("Vrep res: ", res1, res2, res3)
    return motorLeftHandle,motorRightHandle


def propagate(clientID,motorLeftHandle,motorRightHandle, omega1, omega2):
    #### Set the linear velocity of 2 wheels

    vrep.simxSetJointTargetVelocity(clientID,
                                    motorLeftHandle,
                                    omega1, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID,
                                    motorRightHandle,
                                    omega2, vrep.simx_opmode_oneshot)
clientID=initVrep()
motorLeftHandle,motorRightHandle=setVrepHandles(objectNames,clientID, handleNameSuffix = "")
propagate(clientID,motorLeftHandle,motorRightHandle, 1, 1)