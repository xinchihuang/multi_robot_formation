import random
import numpy as np
import pandas as pd
random.seed(88)
# data_x=pd.DataFrame({"a":random.sample(range(100),10),"b":random.sample(range(100),10),"c":random.sample(range(100),10)})
# data_y=pd.Series(random.sample(range(200),10))
# data_y.name="y"
data=pd.read_csv("/home/xinchi/catkin_ws/pose_record.csv",header=None)
data_camera=data.iloc[:, 3:]
data_x=data.iloc[:, 0]
data_y=data.iloc[:, 2]
data_z=data.iloc[:, 1]
print(data_y)
from scipy.optimize import curve_fit
def transform_x(data,a1,a2,a3,t=0):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
def transform_y(data,a1,a2,a3,t=-1.28):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
def transform_z(data,a1,a2,a3,t=0):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
res_x=curve_fit(transform_x,data_camera,data_x,method="lm")[0]
res_y=curve_fit(transform_y,data_camera,data_y,method="lm")[0]
res_z=curve_fit(transform_z,data_camera,data_z,method="lm")[0]
matrix=[res_x,res_y,res_z,[0,0,0,1]]
np.array(matrix)
np.savetxt("params.csv",matrix,delimiter=",")


