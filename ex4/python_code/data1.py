'''
This file is an extension for nnl.py
Consists of solutions from ex4data1.txt
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import scipy.optimize as opt
from scipy.io import loadmat
from nnl import *

# -- reading the input file.
data = read_data('ex4data1.mat')
X = data['X']
y = data['y']

# -- reading weights.
weights = read_data('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# -- feedforward and cost function
#20x20 Input Images of Digits
input_layer_size  = 400
# 25 hidden units
hidden_layer_size = 25
# number of units
num_labels = 10

# unroll parameters
nn_params = np.append(Theta1.flatten(),Theta2.flatten())

# Weight regularization parameter (we set this to 0 here)
lamb = 0
J,_ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print('Unregularized cost at parameters (loaded from ex4weights): ', J,'\n')

lamb = 3
J,_ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print('Regularied cost at parameters (loaded from ex4weights): ', J,'\n')

# random initialization of parameters for breaking the symmetry.
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(),initial_Theta2.flatten())

# checking the implementation of backpropogation
print('Gradient checking for unregularized NN','\n')
checkNNGradients(0)
print('Gradient checking for regularized NN','\n')
checkNNGradients(3)

#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 100}

#  You should also try different values of lambda
lamb_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,hidden_layer_size,num_labels, X, y, lamb)

# Now, costFunction is a function that takes in only one argument (the neural network parameters)
res = optimize.minimize(costFunction,initial_nn_params,jac=True,method='TNC',options=options)

# get the solution of the optimization
nn_params = res.x

Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)

p = predict(Theta1,Theta2,X)
 
print('Training Set Accuracy: %f' % (np.mean(p == y) * 100))
