
# coding: utf-8

# In[34]:

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

def plot3D(data,target,xlabel="Feature 1", ylabel="Feature 2",zlabel="Feature 3",title=""):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', azim=170, elev=50)
    c1 = data[target == 1]
    c0 = data[target == 0]
    ax.scatter(c1[:,0],c1[:,1],c1[:,2], c='g',alpha=0.4, marker='o')    
    ax.scatter(c0[:,0],c0[:,1],c0[:,2], c='r',alpha=0.2, marker='o')    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend(['mem', 'non-mem'])
    plt.title(title)
    plt.show()


# In[35]:

big_path = "/home/rpezoa/experiment_data/big_2+/features/"
big_target_path = "/home/rpezoa/experiment_data/big_2+/labels/"
X  = np.load(big_path + "2+_1_rgb.npy")
y  = np.load(big_target_path + "2+_1.npy")
print(X.shape, y.shape)


# In[36]:

plot3D(X,y,title="RGB features")


# In[ ]:



