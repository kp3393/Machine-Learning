'''
This file is an extension for regression.py
Consists of solutions for single variable input data from ex1data1.txt
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from regression import *

# -- reading the input file.
X,y = read_data('ex1data1.txt',delim=',')

# -- adding a column of ones to the feature array for theta 0
X_appended = np.append(np.ones((np.shape(y)[0],1)),X,1)

# -- plot input data
routine_2dplot([X],[y],[''],'Population of City in 10,000s','Profit in $10,000s',['rx'],'','Fig1:Scatter_plot_of_training_data.png')

# -- compute cost for 
cost_funct = computeCost(np.zeros((2,1)),X_appended,y)

# -- gradient descent
gdsv_to,gdsv_th,gdsv_jh = gradientDescent(np.zeros((2,1)),X_appended,y,0.01,1500)

# -- plot regression vs input data
routine_2dplot([X,X],[y,np.matmul(X_appended,gdsv_to)[:,0]],['Training data','Linear regression'],'Population of City in 10,000s','Profit in $10,000s',['rx','-'],'','Fig2:Scatter_plot_of_training_data_linear_regression.png')

# -- surface and contour plot
theta0_values = np.linspace(-10, 10, 100)
theta1_values = np.linspace(-1, 4, 100)
xx, yy = np.meshgrid(theta0_values, theta1_values, indexing='xy')
J_vals = np.zeros((theta0_values.size,theta1_values.size))

# find out those J values
for (i,j),v in np.ndenumerate(J_vals):
    # remember our theta array is 2x1
    t = np.array([xx[i,j],yy[i,j]]).reshape(2,1)
    J_vals[i,j] = computeCost(t,X_appended,y)

# -- SURFACE PLOT
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_surface(xx, yy, J_vals, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax1.set_zlabel('Cost')
ax1.set_zlim(J_vals.min(),J_vals.max())
ax1.view_init(elev=15, azim=230)
ax1.set_xlabel(r'$\theta_0$')
ax1.set_ylabel(r'$\theta_1$')
plt.savefig('Fig3:surface_plot.png')

# -- Contour plot
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.contour(xx,yy,J_vals,np.logspace(-2, 3, 20),cmap=plt.cm.jet)
ax2.scatter(gdsv_to[0],gdsv_to[1], c='r')
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'$\theta_1$')
plt.savefig('Fig4:contour_plot.png')