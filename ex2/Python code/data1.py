'''
This file is an extension for logistic_regression.py
Consists of solutions from ex2data1.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from logistic_regression import *

# -- reading the input file.
X,y = read_data('ex2data1.txt',delim=',')

# -- find indices of Positive (y = 1) and Negative values (y = 0)
pos = np.where(y==1)[0]
neg = np.where(y==0)[0]
# -- scatter plot of the input data
# routine_2dplot([X[pos,0],X[neg,0]],[X[pos,1],X[neg,1]],['Admitted','Not admitted'],'Exam 1 score','Exam 2 score',['k+','ko'],'','Fig1:Scatter_plot_of_training_set.png')

# -- sigmoid function
# sigmoid(0)

# -- cost function and gradient
X_append = np.append(np.ones((np.shape(y)[0],1)),X,1)
initial_theta = np.zeros((3,1))
cost,gradient = costFunction(initial_theta,X_append,y)
print('Cost at initial theta (zeros):\n', cost,'\n')
print('Gradient at initial theta (zeros):\n',gradient,'\n')

# -- advanced optimization techniques. Finding the correct theta.

# -- NOTE on flatten() function: Unfortunately scipy’s fmin_tnc doesn’t work
# -- well with column or row vector. It expects the parameters to be in an array format.
# -- The flatten() function reduces a column or row vector into array format.
# -- messages = 0 suppress onscreen messages in command prompt
result = opt.fmin_tnc(func=costFunction, x0=initial_theta.flatten(),messages = 0, args=(X_append, y.flatten()))
optimised_theta = result[0].reshape(result[0].size,1)
print('Thetas found by fmin_tnc function: ', optimised_theta,'\n')

# -- cost function at optimised theta
cost_opt,gradient_opt = costFunction(optimised_theta,X_append,y)
print('Cost at theta found by scipy fmin_tnc = \n',cost_opt,'\n')

# -- plotting of decision boundary
y_db = (1/optimised_theta[2])*(-(X_append[:,0]*optimised_theta[0])-(X_append[:,1]*optimised_theta[1]))
# routine_2dplot([X[pos,0],X[neg,0],X_append[:,1]],[X[pos,1],X[neg,1],y_db],['Admitted','Not admitted','Decision Boundary'],'Exam 1 score','Exam 2 score',['k+','ko','-'],'','Fig2:Training_data_with_decision_boundary.png')

# --  Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
prob = sigmoid(np.matmul(np.array([1,45,85]).reshape(1,3),optimised_theta))
print('For a student with scores 45 and 85, we predict an admission probability of \n', prob,'\n')

# -- Compute accuracy on our training set
p = predict(optimised_theta,X_append)
print('Train Accuracy: \n',np.mean(p==y)*100,'\n')
