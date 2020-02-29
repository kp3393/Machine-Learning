'''
This file implements the ex1 of Andrew Ng's MAchine learning course using Python.
1. Read the input files which are seperated by comma.
2. Normal 2D plot
3. Cost function 
4. Gradient descent
5. Feature Normalization
5. Normal Equation
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def computeCost(theta,X,y):
    '''
    Computes the cost J(theta) for given values of theta and input X
    
    :param theta : numpy array of shape(nos of parameters,1). value of parameters theta.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset
    
    :returns : cost function
    '''
    # length of input signal
    m = np.shape(y)[0]
    J = 0
    # calculating cost function
    J = (0.5/m)*np.sum(np.square((np.matmul(X,theta)-y)))
    return J

def gradientDescent(theta,X,y,alpha,num_iters):
    '''
    Computes optimized values for parameters theta
    
    :param theta : numpy array of shape(nos of parameters,1). value of parameters theta.
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param alpha : learning rate
    :param num_iters : number of iterations for which the optimsation should run
    
    :returns : theta, theta_history, J_history
    theta : array of shape (nos of theta,1) consisting of optimized values of the parameters
    theta_history : array of shape (num_iters, nos of theta) consisting of theta values for each iteration
    J_history :  array of shape (num_iters,1) consisting of cost function for each set of theta value
    '''
    # length of input signal
    m = np.shape(y)[0]
    temp = np.zeros((np.shape(theta)[0],1))
    theta_history = np.zeros((num_iters,np.shape(theta)[0]))
    J_history = np.zeros((num_iters,1))
    
    for iter in range(num_iters):
        for thetas in range(np.shape(theta)[0]):
            temp[thetas,0] = theta[thetas,0] - ((alpha/m) * np.matmul(np.transpose((np.matmul(X,theta)-y)),X[:,thetas]))
        theta = temp
        theta_history[iter,:] = np.transpose(temp)
        J_history[iter,0] = computeCost(theta,X,y)
    
    return theta, theta_history, J_history

def featureNormalize(X):
    '''
    Feature scaling for faster convergence
    
    :param X : array of shape(sample size, number of FEATURES). features of the given dataset.
    
    :returns : X_norm, mu, sigma
    X_norm : array of shape(sample size, number of parameters). Mean normalization of values. (x-mu)/sigma
    mu : array of shape(number of FEATURES,1). Mean of each feature set.
    sigma : array of shape(number of FEATURES,1). Standard deviation of each feature set.
    '''
    X_norm = np.zeros((np.shape(X)))
    mu = np.zeros((np.shape(X)[1]))
    sigma = np.zeros((np.shape(X)[1]))
    
    for column in range(np.shape(X_norm)[1]):
        mu[column] = np.mean(X[:,column])
        sigma[column] = np.std(X[:,column])
        for row in range(np.shape(X_norm)[0]):
            X_norm[row,column] = (X[row,column]-mu[column])/(sigma[column])
    
    return X_norm, mu, sigma
    
def normalEqn(X,y):
    '''
    Computes closed form solution to normal linear regression
    
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0
    :param y : array of shape(sample size, 1).outputs of the given dataset
    
    :returns : theta
    theta : array of shape(number of parameters,1). Optimised parameters for linear regression
    '''
    theta = np.zeros((np.shape(X)[1]))
    
    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)
    return theta
''' ----#### END OF FUNCTIONS ####---- '''

