'''
This file implements the ex3 (Multi-class Classification and Neural Network ) of Andrew Ng's Machine learning course using Python.
1. reads the data from .mat file
2. sigmoid function 
3. lrCostFunction to calculate regularized cost function
4. oneVsAll to multi class classification
5. predictOneVsAll for predicting how good is the implementation
6. predict for neural network
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from scipy.io import loadmat

''' ----#### FUNCTIONS ####---- '''
def read_data(fname, delim = ',', header=False):
    '''
    Get columns of data from given file names. The order of row is respected.
    
    :param fnames : file name which needs to read
    :param delim :  default delimited is tab
    
    :returns :
    X : array of features
    y : array of output values 
    '''
    data = loadmat(fname)
    return data

def sigmoid(z):
    '''
    Computes sigmoid of z. g(z) = 1/(1+exp(-z))
    
    :param z : Vector, array or matrices. Input values for which sigmoid function needs to be calculated.
    '''
    g = np.zeros((np.shape(z)))
    g = np.dot(1,(1+np.exp(-z))**-1)
    
    return g    

def lrCostFunction(theta, X, y, lamb):
    '''
    Computes cost and gradient for logistic regression with regularization. Computes the cost of using theta as the parameter
    for regularized logistic regression and the gradient of the cost w.r.t. parameters.
    
    :param theta : numpy array of shape(nos of parameters,1). value of parameters theta.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param lamb : integer. regularization parameter.
    
    returns : J, grad
    J : array of shape (num_iters,1) consisting of Regularied cost function for each set of theta value.
    grad : numpy array of shape(nos of parameters,1). gradient of cost w.r.t. parameters.
    '''
    # -- WHEN USING fmin_tnc for finding optimised values of a function the inputs are a flattened arrays. This does not matches with our data format.
    # -- thus the following two lines make sure that there is a consistency is our data.
    theta = theta.reshape(theta.size,1)
    y = y.reshape(y.size,1)
    
    # number of training examples
    m = np.shape(y)[0]
    
    # Initialise cost function
    J = 0
    
    # Initialise gradient function
    grad = np.zeros(np.shape(theta))
    
    J = (-1/m)*((np.matmul(np.transpose(y),np.log(sigmoid(np.matmul(X,theta)))))+(np.matmul(np.transpose(1-y),np.log(1-sigmoid(np.matmul(X,theta)))))) + (lamb/(2*m))*(np.sum(theta[1:,0]**2))
    grad[0,0] = (1/m)*np.transpose(np.matmul(np.transpose((sigmoid(np.matmul(X,theta))-y)),X[:,0]))
    add = (lamb/m)*(theta[1:,0])
    grad[1:,0:] = (1/m)*np.transpose(np.matmul(np.transpose((sigmoid(np.matmul(X,theta))-y)),X[:,1:]))+add.reshape(add.size,1)
    return J,grad
    
def oneVsAll(X, y, num_labels, lamb):
    '''
    trains multiple logistic regression classifiers and returns all the classifiers 
    in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier
    for label i.
    
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param num_labels : array of shape(number of labels, 1). number of labels for which logistic regression needs to be trained
    :param lamb : integer. regularization parameter.
    
    returns : all_theta
    all_theta : array of shape(number of labels, number of columns in x). Optimized theta values for each classifier.
    '''
    # number of training examples
    m = np.shape(y)[0]
    # number of features
    n = np.shape(X)[1]
    
    all_theta = np.zeros((len(num_labels),n))
    
    for c in range(1,len(num_labels)+1):
        # setting initial theta to zeros
        initial_theta = np.zeros((n,1))
        result = opt.fmin_tnc(func=lrCostFunction, x0=initial_theta.flatten(),messages = 0, args=(X, (y==c).flatten(),lamb))
        all_theta[c-1,:] = result[0]
    return all_theta
    
def predictoneVsAll(all_theta,X):
    '''
    PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = size(all_theta, 1).
    
    :param all_theta : array of shape(number of labels, number of columns in X). Optimized theta values for each classifier.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    
    returns : p
    p : array of shape (sample size,1). Vector of predictions for each example in matrix X.
    '''
    # number of training examples
    m = np.shape(X)[0]
    # number of features
    n = np.shape(X)[1]
    # num_label. Number of labels
    k = np.shape(all_theta)[0]
    # temp to save intermediate results
    temp = np.zeros((m,k))
    temp = sigmoid(np.matmul(X,np.transpose(all_theta)))
    # find the index of the maximum postion in the table
    p = np.argmax(temp,axis = 1).reshape((np.shape(temp)[0]),1)
    # since indexing in python startes from 0 we need to add one to get the correct output.
    return p+1
    
def predict(Theta1, Theta2, X):
    '''
    Predicts the label of an input given a trained neural network.
    
    :param Theta1 : a column array
    :param Theta2 : a column array
    : param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    
    returns: p
    p : array of shape (sample size,1). outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)
    '''
    # number of training examples
    m = np.shape(X)[0]
    # number of features
    n = np.shape(X)[1]
    # output p
    p = np.zeros((m,1))
    
    # repeat it for each example in X
    for row in range(m):
        # first layer of size ((no_of_features) x 1)
        a1 = np.zeros((n,1))
        # second layer of size ((number of rows in theta 1 + 1) X 1).Add one to take into consideration one.
        a2 = np.zeros((np.shape(Theta1)[0]+1,1))
        # output layer ((number of rows in theta 2) X 1)
        a3 = np.zeros((np.shape(Theta2)[0],1))
        # adding 1 to first cell of layer 2
        a2[0,0] = 1
        #layer 1: input layer
        a1 = np.transpose(X[row,:])
        #layer 2: hidden layer
        a2[1:,0] = sigmoid(np.matmul(Theta1,a1))
        #layer 3: output layer
        a3 = sigmoid(np.matmul(Theta2,a2))
        p[row,0] = np.argmax(a3,axis=0)+1
    return p
         