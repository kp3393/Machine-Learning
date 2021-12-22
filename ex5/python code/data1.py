'''
This script is in supplement with bias_vs_variance.py.
The script implements the results for ex5data1.py.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bias_vs_variance import *

# training set data
X = data['X']
y = data['y']
# cross validation set data
Xval = data['Xval']
yval = data['yval']
# test set data
Xtest = data['Xtest']
ytest = data['ytest']

# size of training set
m_train = X.shape[0]
m_val = Xval.shape[0]
m_test = Xtest.shape[0]

# -- plot the training data
routine_2dplot([X],[y],[''],'Change in water level (x)','Water flowing out of the dam (y)','x','Training set','Fig1.png')

# -- regularized cost function
theta = np.array([[1],[1]])
J,grad = linearRegCostFunction(np.append(np.ones((m_train,1)),X,1),y,theta,1)
print('1. Cost at theta = [1;1] : %f'%J,'\n')
print('2. Gradient at theta = [1;1] : ',grad,'\n')

# -- training linear regression with lambda = 0
lamb = 0
theta = trainLinearReg(np.append(np.ones((m_train,1)),X,1),y)
reg_fit = np.matmul(np.append(np.ones((m_train,1)),X,1),theta)
# -- plot the linear fit
routine_2dplot([X,X],[y,reg_fit],['Training data','Linear reg'],'Change in water level (x)','Water flowing out of the dam (y)',['x','--'],'Linear fit','Fig2.png')

# -- learning curve
X_append = np.append(np.ones((m_train,1)),X,1)
Xval_append = np.append(np.ones((m_val,1)),Xval,1)
error_train,error_val = learningCurve(X_append,y,Xval_append,yval,lamb = 0)

# -- plot the learning curve
routine_2dplot([np.arange(1,m_train+1),np.arange(1,m_train+1)],[error_train,error_val],['Training error','Cross validation error'],'Number of training example','Error',['',''],'Learning curve for linear regression','Fig3.png')

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m_train):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
print('\n')

# -- Learning polynomial regression
p = 8
# -- Map X into polynomial features
X_poly = polyFeatures(X,p)
# -- Normalise
X_poly,mu,sigma = featureNormalize(X_poly)
# -- concatenate X_poly
X_poly = np.append(np.ones((m_train,1)),X_poly,1)

# -- map X_poly_test and normalise (using mu and sigma)
X_poly_test = polyFeatures(Xtest,p)
# -- normalise
X_poly_test -= mu
X_poly_test /= sigma
# -- append
X_poly_test = np.append(np.ones((m_test,1)),X_poly_test,1)

# -- map X_poly_val and normalise (using mu and sigma)
X_poly_val = polyFeatures(Xval,p)
# -- normalise
X_poly_val -= mu
X_poly_val /= sigma
# -- append
X_poly_val = np.append(np.ones((m_val,1)),X_poly_val,1)

# -- normalized training example test
print('Normalized Training Example 1:','\n',X_poly[0,:],'\n')

# -- Learning polynomial regression
lambdaPlot(0,X,p,X_poly,y,mu,sigma,X_poly_val,yval,4)
lambdaPlot(100,X,p,X_poly,y,mu,sigma,X_poly_val,yval,6)

# -- selecting lambda using a cross validation set
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
routine_2dplot([lambda_vec,lambda_vec],[error_train,error_val],['Train','Cross Validation'],'lambda','Error',['-o','-o'],'Lambda selection using cross validation set','Fig8.png')

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]),'\n')
    
theta = trainLinearReg(X_poly, y, 3)
J_test,_ = linearRegCostFunction(X_poly_test,ytest,theta,lamb = 0)
print('Test error for lambda = 3 is %f '%J_test)