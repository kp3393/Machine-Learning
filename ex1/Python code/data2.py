'''
This file is an extension for regression.py
Consists of solutions for single variable input data from ex1data1.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from regression import *

# -- reading the input file. Remember the input file is in string.
X,y = read_data('ex1data2.txt',delim=',')

# -- feature normalization
X,mu,sigma = featureNormalize(X)

# -- adding a column of ones to the feature array for theta 0
X_appended = np.append(np.ones((np.shape(y)[0],1)),X,1)

# -- gradient descent for multiple variable
gdmv_to,gdmv_th,gdmv_jh = gradientDescent(np.zeros((3,1)),X_appended,y,0.1,400)

# -- plotting convergence for different learning rates
J_history,x_axis,alpha = [[] for i in range(3)]
alpha = [1,0.3,0.1,0.03,0.01]
for i in alpha:
    J_history.append(gradientDescent(np.zeros((3,1)),X_appended,y,i,400)[2]) 
    x_axis.append(np.arange(400))

routine_2dplot(x_axis,J_history,['1','0.3','0.1','0.03','0.01'],'Iterations','Cost functions',['-','-','-','-','-','-'],'','Fig4:costFunc_vs_numIters.png')

# Normal equation
# again read the data because we do not need feature normalization
X,y = read_data('ex1data2.txt',delim=',')
X_appended = np.append(np.ones((np.shape(y)[0],1)),X,1)
normal_theta = normalEqn(X_appended,y)