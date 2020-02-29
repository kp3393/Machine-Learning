'''
This file implements the ex2 of Andrew Ng's Machine learning course using Python.
1. Read the input files which are seperated by comma.
2. Normal 2D plot

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

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
    data = np.loadtxt(fname,delimiter=delim)
    X = data[:,0:np.shape(data)[1]-1]
    y = data[:,np.shape(data)[1]-1:]
    return X,y
    
def routine_2dplot(xvalues,yvalues,legends,xlabels,ylabels,markers,title,imgname):
    '''
    Creates a 2d plot of the input data. Single graph or multiple graphs in one single graph.
    
    :param xvalues : list consisting of x values of the graph to be plotted
    :param yvalues : list consisting of corresponding y values of the graph to be plotted
    :param legends : list of legends for the plotted graphs
    :param xlabels : string for x axis label
    :param ylabels : string for y axis label
    :param title : string for graph title
    :param imgname : string for filename. save image with this name as a png file in the present directory with a dpi = 600.
    
    output : saves a 2D plot in the working directory with the name 'imgname.png'
    '''
    for i in range(len(xvalues)):
        if legends[i] != '':
            plt.plot(xvalues[i],yvalues[i],markers[i],label = legends[i])
            plt.legend()
        else:
            plt.plot(xvalues[i],yvalues[i],markers[i])
    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(title)
    plt.savefig(imgname,format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.clf()

def sigmoid(z):
    '''
    Computes sigmoid of z. g(z) = 1/(1+exp(-z))
    
    :param z : Vector, array or matrices. Input values for which sigmoid function needs to be calculated.
    '''
    g = np.zeros((np.shape(z)))
    g = np.dot(1,(1+np.exp(-z))**-1)
    
    return g
    
def costFunction(theta, X, y):
    '''
    Computes cost and gradient for logistic regression.

    :param theta : numpy array of shape(nos of parameters,1). value of parameters theta.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset

    :returns : J, grad
    J : array of shape (num_iters,1) consisting of cost function for each set of theta value.
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
    # Initialise gradient
    grad = np.zeros(np.shape(theta))
    J = (-1/m)*((np.matmul(np.transpose(y),np.log(sigmoid(np.matmul(X,theta)))))+(np.matmul(np.transpose(1-y),np.log(1-sigmoid(np.matmul(X,theta))))))
    grad = (1/m)*np.transpose(np.matmul(np.transpose((sigmoid(np.matmul(X,theta))-y)),X))
    return J,grad
    
def predict(theta,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters theta using a threshold at 0.5.
    if sigmoid(X*theta)>=0.5, predict 1 or else 0.
    
    :param theta : numpy array of shape(nos of parameters,1). value of parameters theta.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    
    :returns : p
    predicted output array with values 0 or 1
    '''
    # number of training examples
    m = np.shape(X)[0]
    p = np.zeros((m,1))
    
    # find where sigmoid function is greater than 0.5 for given input
    pos = np.where(sigmoid(np.matmul(X,theta))>=0.5)[0]
    p[pos,0] = 1
    
    return(p)
    
def mapFeature(X1,X2):
    '''
    Feature mapping function to polynomial features. Maps two input features to quadratic features.
    Both input features must be of same size.
    
    :param X1 : numpy array of shape(no of samples,1).
    :param X2 : numpy array of shape(no of samples,1).
    
    :returns : out
    Returns a new feature array with more features, comprising of 
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc.
    '''
    degree = 6
    out = np.ones((np.size(X1),1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out = np.append(out,np.multiply((X1**(i-j)),X2**j),1)
    return out
    
def costFunctionReg(theta, X, y, lamb):
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