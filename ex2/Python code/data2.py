'''
This file is an extension for logistic_regression.py
Consists of solutions from ex2data2.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from logistic_regression import *

# -- reading the input file.
X,y = read_data('ex2data2.txt',delim=',')

# -- find indices of Positive (y = 1) and Negative values (y = 0)
pos = np.where(y==1)[0]
neg = np.where(y==0)[0]

# -- scatter plot of the input data
# routine_2dplot([X[pos,0],X[neg,0]],[X[pos,1],X[neg,1]],['y=1','y=0'],'Microchip Test 1','Microchip Test 2',['k+','ko'],'','Fig3:plot_of_training_data.png')

X = mapFeature(X[:,0].reshape(X[:,0].shape[0],1),X[:,1].reshape(X[:,1].shape[0],1))

#-- Initialize fitting parameters
initial_theta = np.zeros((np.shape(X)[1],1))

# -- Set regularization parameter lambda to 1
# lamb = 0
# lamb = 1
# lamb = 10
# lamb = 100
# lamb = 1000
# -- Compute and display initial cost and gradient for regularized logistic regression
cost, grad = costFunctionReg(initial_theta, X, y, lamb)
print('Cost at initial theta (zeros):\n', cost,'\n')

# -- advanced optimization techniques. Finding the correct theta.

# -- NOTE on flatten() function: Unfortunately scipy’s fmin_tnc doesn’t work
# -- well with column or row vector. It expects the parameters to be in an array format.
# -- The flatten() function reduces a column or row vector into array format.
# -- messages = 0 suppress onscreen messages in command prompt
result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta.flatten(),messages = 0, args=(X, y.flatten(),lamb))
optimised_theta = result[0].reshape(result[0].size,1)
print('Thetas found by fmin_tnc function: ', optimised_theta,'\n')

# -- cost function at optimised theta
cost_opt,gradient_opt = costFunction(optimised_theta,X,y)
print('Cost at theta found by scipy fmin_tnc = \n',cost_opt,'\n')

# -- Compute accuracy on our training set
p = predict(optimised_theta,X)
p = np.mean(p==y)*100
print('Train Accuracy: \n',p,'\n')

# -- surface and contour plot
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((u.size,v.size))
print(np.shape(np.array(u[0]).reshape(1,1)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.matmul(mapFeature(np.array([u[i]]).reshape(1,1),np.array([v[j]]).reshape(1,1)),optimised_theta)

plt.plot(X[pos,1],X[pos,2],'k+')
plt.plot(X[neg,1],X[neg,2],'ko')
plt.contour(u, v, z, levels = 1)
plt.legend(['y=0','y=1','decision boundary'])
plt.title('lambda = %f' % lamb+' & accuracy = %f'%p)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.savefig('Fig4:decision_boundary_with_lambda %f.png'%lamb)
plt.show()