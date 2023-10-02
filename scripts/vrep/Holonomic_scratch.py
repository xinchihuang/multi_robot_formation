import sim as vrep
import time
import math
# Connect to V-REP
vrep.simxFinish(-1)  # Close all open connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print('Connected to remote API server')
else:
    print('Failed to connect to remote API server')
    exit(1)
vrep.simxSynchronous(clientID, True)
vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)


# Get object handles
res, omniRob_handle = vrep.simxGetObjectHandle(clientID, 'Omnirob', vrep.simx_opmode_blocking)
# vrep.simxSetObjectPosition(
#     clientID, omniRob_handle, -1, [-6, 0, 0.5], vrep.simx_opmode_oneshot
# )
# vrep.simxSetObjectOrientation(
#     clientID, omniRob_handle, -1, [0, 0, math.pi/2], vrep.simx_opmode_oneshot
# )
_, omniRob_RL_handle = vrep.simxGetObjectHandle(clientID, 'Omnirob_RLwheel_motor', vrep.simx_opmode_oneshot_wait)
_, omniRob_RR_handle = vrep.simxGetObjectHandle(clientID, 'Omnirob_RRwheel_motor', vrep.simx_opmode_oneshot_wait)
_, omniRob_FL_handle = vrep.simxGetObjectHandle(clientID, 'Omnirob_FLwheel_motor', vrep.simx_opmode_oneshot_wait)
_, omniRob_FR_handle = vrep.simxGetObjectHandle(clientID, 'Omnirob_FRwheel_motor', vrep.simx_opmode_oneshot_wait)

print(res,omniRob_handle,omniRob_RL_handle,omniRob_RR_handle,omniRob_FL_handle,omniRob_FR_handle)
# Control the robot
if res == vrep.simx_return_ok:
    vfl, vfr, vrl, vrr=0.1,0,0,0.1
    print(vfl, vfr, vrl, vrr)
    vrep.simxSetJointTargetVelocity(clientID, omniRob_FL_handle, -vfl, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID, omniRob_FR_handle, vfr, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID, omniRob_RL_handle, -vrl, vrep.simx_opmode_oneshot)
    vrep.simxSetJointTargetVelocity(clientID, omniRob_RR_handle, vrr, vrep.simx_opmode_oneshot)
    # vrep.simxSetJointTargetVelocity(clientID, omniRob_RL_handle, -0.1, vrep.simx_opmode_oneshot)
    # vrep.simxSetJointTargetVelocity(clientID, omniRob_RR_handle, 0.1, vrep.simx_opmode_oneshot)
    # vrep.simxSetJointTargetVelocity(clientID, omniRob_FL_handle, -0.1, vrep.simx_opmode_oneshot)
    # vrep.simxSetJointTargetVelocity(clientID, omniRob_FR_handle, 0.1, vrep.simx_opmode_oneshot)
    print("running")
else:
    print('Failed to get omniRob handle')
    exit(1)

# End simulation and close connection
# vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
# vrep.simxFinish(clientID)
