import random
import numpy as np
import pandas as pd
random.seed(88)
data_x=pd.DataFrame({"a":random.sample(range(100),10),"b":random.sample(range(100),10),"c":random.sample(range(100),10)})
data_y=pd.Series(random.sample(range(200),10))
data_y.name="y"

from scipy.optimize import curve_fit
def transform_x(data,a1,a2,a3,t):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
def transform_y(data,a1,a2,a3,t):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
def transform_z(data,a1,a2,a3,t):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    return a1*x+a2*y+a3*z+t
res=curve_fit(transform_x,data_x,data_y,method="lm") #p0 设置初始值, maxfev=10000 设置最大迭代次数
print(res)


import matplotlib.pyplot as plt
k,b,_,_=res[0]
print(k,b)
plt.figure(figsize=(8,6))
plt.scatter(data_x.iloc[:,0],data_y.iloc[:],color="red",label="Sample Point",linewidth=3)
x = np.linspace(0,100,1000)
y = k * x + b
plt.plot(x,y,color="orange",label="Fitting Line",linewidth=2)
plt.legend()
plt.show()