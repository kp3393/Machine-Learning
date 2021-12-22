'''
This file is an extension for logistic_regression.py
Consists of solutions from ex2data1.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from mcc import *


# -- reading the input file.
data = read_data('ex3data1.mat')
X = data['X']
y = data['y']

# -- Verifying for cost function implementation
theta_t = np.array([[-2],[-1],[1],[2]])
X_t = np.append(np.ones((5,1)),(np.transpose(np.arange(1,16)).reshape((5,3),order = 'F')/10),1)
y_t = np.array([[1],[0],[1],[0],[1]])
lambda_t = 3
J,grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost: %f | Expected cost: 2.534819\n'%J)
print('Gradients:\n',grad,'\n')
print('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003','\n')

# multi-class classification
num_labels = np.arange(1,11)
lamb = 0.1
X_append = np.append(np.ones((np.shape(y)[0],1)),X,1)
all_theta = oneVsAll(X_append, y, num_labels, lamb)
p = predictoneVsAll(all_theta,X_append)
print('Train Accuracy: \n',np.mean(p==y)*100,'\n')

# feedforward propogation and prediction
weights = read_data('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

pred = predict(Theta1, Theta2, X_append)
print('Train Accuracy: \n',np.mean(pred==y)*100,'\n')
