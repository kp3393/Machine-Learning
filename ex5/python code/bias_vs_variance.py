'''
This python script implements exerice 5 of Andrew Ng's Machine learning course. Problem is explained in the pdf ex5.pdf.
Functions implemented
1. read_data
2. routine_2dplot
3. linearRegCostFunction
4. trainLinearReg
5. learningCurve
6. polyFeatures
7. featureNormalize
8. polyFit
9. lambdaPlot
10. validationCurve 
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.io import loadmat

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

def linearRegCostFunction(X,y,theta,lamb):
    '''
    Compute cost and gradient for regularized linear regression with multiple variables.
    
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param theta : array of shape (number of features+1,1). parameters of fitting function.
    :param lamb : float. regularization parameter.
    
    returns : J and grad
    J : cost function value for the given dataset
    grad : array of shape (number of features+1,1). gradient of cost function with respect to given parameters
    '''
    # number of samples(m) and number of features(n)
    m,n = X.shape
    # need to reshape the theta in nx1 because when using optimisation technique, theta dimensions are different.
    theta = theta.reshape(n,1)
    # declaring important variables
    J = 0
    grad = np.zeros(theta.shape)
    # cost function
    J = ((1/(2*m))*(sum(np.square(np.matmul(X,theta)-y)))) + ((lamb/(2*m))*sum(np.square(theta[1:,0])))
    # gradient calculation
    grad[:,0] = ((1/m)*sum(np.multiply(np.matmul(X,theta)-y,X)))
    grad[1:,0] = grad[1:,0] + (lamb/m)*theta[1:,0]
    return J[0],grad
    
def trainLinearReg(X,y,lamb = 0,maxiter = 200):
    '''
    Trains linear regression given a dataset (X,y) and a regularization parameter lamb
    
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0.
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param lamb : float. regularization parameter. Default is 0.
    :param maxiter : maximum number of iteration for optimization parameter. Default is 200.
    
    returns : theta
    theta : array of shape (number of features + 1,1). Optimized paramter for a dataset.
    '''
    # inital value of theta
    initial_theta = np.zeros(X.shape[1])
    
    # creating short hand notation for our costFunction which is to be minimized. 
    # Now costfunction is a function which takes only one argument.
    costFunction = lambda t: linearRegCostFunction(X,y,t,lamb)
    
    # minimise using scipy
    options = {'maxiter':maxiter}
    res = optimize.minimize(costFunction,initial_theta,jac = True, method = 'TNC',options = options)
    return res.x
    
def learningCurve(X, y, Xval, yval, lamb):
    '''
    Generates the train and cross validation set error needed to plot a learning curve.
    
    :param X : array of shape(sample size, number of parameters). features of the given dataset. Column of ones needs to be included as well for theta0
    :param y : array of shape(sample size,1). outputs of the given dataset
    :param Xval : array of shape (m_val,number of paramters) where m_val = sample size of cross validation set. Cross validation set
    :param yval :  array of shape (m_val,1). output for cross validation dataset
    :lamb : float. Regularization parameter 
    
    returns : error_train, error_val
    error_train : a vector of shape m. error_train[i] contains the training error for i examples
    error_val : a vector of shape m. error_val[i] contains the validation error for i training examples
    '''
    m,n = X.shape
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    
    for example in range(1,m+1):
        theta = trainLinearReg(X[:example,:],y[:example,:],lamb)
        error_train[example-1],_ = linearRegCostFunction(X[:example],y[:example],theta,0)
        error_val[example-1],_ = linearRegCostFunction(Xval,yval,theta,0)
    return error_train,error_val
    
def polyFeatures(X,p):
    '''
    Maps X (1D vector) into the p-th power.
    
    param X : array like. A data vector of size m, where m is the number of examples.
    param p : the polynomial power to map the function.
    
    returns : X_poly
    X_poly : A matrix of shape (mxp) where p is the polynomial power and m is the number of samples.
    '''
    X_poly = np.zeros((X.size,p))
    for col in range(1,p+1):
        X_poly[:,col-1] = X[:,0]**col
    return X_poly

def featureNormalize(X):
    '''
    Feature scaling for faster convergence
    
    :param X : array of shape(sample size, number of FEATURES). features of the given dataset.
    
    :returns : X_norm, mu, sigma
    X_norm : array of shape(sample size, number of parameters). Mean normalization of values. (x-mu)/sigma
    mu : array of shape(number of FEATURES,1). Mean of each feature set.
    sigma : array of shape(number of FEATURES,1). Standard deviation of each feature set.
    '''
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma

def polyFit(X,mu,sigma,theta,p):
    '''
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).
    We plot a range slightly bigger than the min and max values of X to get
    an idea of how the fit will vary outside the range of the data points
    
    :param X : array of shape(sample size, number of FEATURES). features of the given dataset.
    :param mu : The mean feature value of the existing data set.
    :param sigma : The standard deviation of the training set
    :param theta : array like. The parameter for the trained polynomial linear regression.
    :param p : int. The polynomial order.
    
    returns : x,x_poly_fit
    x : array like. x axis for x_poly_fit.
    x_poly_fit : array of shape (number of samples,1).
    linear regression for polynomial feature set.
    '''
    # extending the range of x values.
    x = np.arange(np.min(X)-15,np.max(X)+25,0.05).reshape(-1,1)
    
    # creating polynomial features out of it.
    x_poly = polyFeatures(x,p)
    # normalising the featureset
    x_poly -= mu
    x_poly /= sigma
    
    # adding a set of ones
    x_poly = np.append(np.ones((x.shape[0],1)),x_poly,1)
    # linear regression for polynomial feature set
    x_poly_fit = np.matmul(x_poly,theta)
    
    return x,x_poly_fit
    
def lambdaPlot(lamb,X,p,X_poly,y,mu,sigma,X_poly_val,yval,img_num):
    '''
    Plots fitted curve and learning curve for a given set of data.
    
    :param lamb : float. regularization parameter
    :param X : array of shape(sample size, number of FEATURES). without bias column
    :param p : int. The polynomial order.
    :param X_poly : array of shape(mxp) where p is the polynomial power and m is the number of samples
    :param y : array of shape(sample size,1). outputs of the given dataset
    :param mu : The mean feature value of the existing data set.
    :param sigma : The standard deviation of the training set
    :param X_poly_val : array of shape(mxp) where p is the polynomial power and m is the number of samples in validation set
    :param yval : array pf shape(mx1) where m is the number of samples in validation set
    :param img_num : integer. saves the image as Fig img_num for fitting and Fig img_num+1 for learning curve.
    
    returns : two graph is saved in the working directory.
    '''
    m_train = X.shape[0]
    theta = trainLinearReg(X_poly,y,lamb,maxiter = 55)
    # -- To get the idea how the fit will vary outside the range of the data points
    # -- we need to extend the range of X slightly bigger than the min and max values.
    x_poly_xax,x_poly_fit = polyFit(X,mu,sigma,theta,p)
    # -- plotting the polynomial fit
    routine_2dplot([X,x_poly_xax],[y,x_poly_fit],['Training set','Polynomial fit'],'Change in water level (x)','Water flowing out of the dam (y)',['x','--'],'Polynomial Regression Fit (lambda = %f)'%lamb,'Fig%s.png'%str(img_num))
    
    # plotting learning curve
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lamb)
    routine_2dplot([np.arange(1,m_train+1),np.arange(1,m_train+1)],[error_train,error_val],['Training error','Cross validation error'],'Number of training example','Error',['',''],'Polynomial Regression Learning Curve (lambda = %f)'%lamb,'Fig%s.png'%str(img_num+1))
    # print the values on the command prompt
    print('Polynomial regression for lambda = %f'%lamb,'\n')
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m_train):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))
    print('\n')
    
def validationCurve(X,y,Xval,yval):
    '''
    Generate the train and validation errors needed to plot a validation 
    curve that we can use to select lamb.
    
    :param X : array of shape(sample size, number of FEATURES). without bias column
    :param y : array of shape(sample size,1). outputs of the given dataset
    :param Xval : array of shape (m_val,number of paramters) where m_val = sample size of cross validation set. Cross validation set
    :param yval :  array of shape (m_val,1). output for cross validation dataset
    
    returns : lambda_vec, error_train, error_val
    lambda_vec : list.  The values of the regularization parameters which were used in cross validation
    error_train : list. The training error computed at each value for the regularization parameter
    error_val : list. The validation error computed at each value for the regularization parameter
    '''
    # Selected values of lambda
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    
    # error train will store the values of training cost function for different lambdas.
    error_train = np.zeros(len(lambda_vec))
    # error val will store the values of validation cost function for different lambdas.
    error_val = np.zeros(len(lambda_vec))
    
    for i in range(len(lambda_vec)):
        theta = trainLinearReg(X,y,lamb = lambda_vec[i])
        error_train[i],_ = linearRegCostFunction(X,y,theta,lamb = 0)
        error_val[i],_ = linearRegCostFunction(Xval,yval,theta,lamb = 0)
    
    return lambda_vec,error_train,error_val
    
    