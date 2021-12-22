'''
This python script implements exerice 8 of Andrew Ng's Machine learning course. Problem is explained in the pdf ex8.pdf.
Functions implemented
1. read_data
2. routine_2dplot
3. estimateGaussian
4. multivariateGaussian
5. visualizeFit
6. selectThreshold
7. cofiCostFunc
8. checkCostFunction
9. loadMovieList
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from scipy.io import loadmat
from mpl_toolkits import mplot3d

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
    
def estimateGaussian(X):
    '''
    Estimates the parameters of Gaussian distribution using the data in X
    
    :param X : array of shape (m,n). Input dataset with m examples and n features.
    
    :returns : mu, sigma2
    mu : array of shape (n,1). mean of each feature in X.
    sigma2 : array of shape (n,1). variance of each feature in dataset.
    '''
    # -- useful variables
    m,n = X.shape
    
    mu = np.mean(X,axis=0)
    sigma2 = np.var(X,axis=0)
    
    return mu, sigma2
    
def multivariateGaussian(X, mu, Sigma2):
    '''
    Computes the probability density function of the multivariate gaussian distribution.
    
    : param X : array of shape (m,n). Input dataset with m examples and n features.
    : param mu : A vector of shape (n,) contains the means for each dimension (feature).
    : param Sigma2 :  Either a vector of shape (n,) containing the variances of independent features
                    (i.e. it is the diagonal of the correlation matrix), or the full
                    correlation matrix of shape (n x n) which can represent dependent features.
    
    : returns : p 
    p :  A vector of shape (m,) which contains the computed probabilities at each of the
         provided examples.
    '''
    k = mu.size
    # -- if sigma is a vector of shape n then we need to make it a diagonal matrix.
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)
    
    X = X - mu
    # -- implementing the expression for multivariate gaussian from lecture slides page 26
    p = ((2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)) * (np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1)))
    return p

def visualizeFit(X, mu, Sigma2):
    '''
    Visualize the dataset and its estimated distribution.
    This visualization shows you the  probability density function of the Gaussian distribution.
    Each example has a location (x1, x2) that depends on its feature values.
    
    : param X : array of shape (m,2). We have m examples of 2 features or 2 dimensions.
    : param mu : vector of shape (n,). mean of each dimension.
    : param Sigma2 :  Either a vector of shape (n,) containing the variances of independent features
                    (i.e. it is the diagonal of the correlation matrix), or the full
                    correlation matrix of shape (n x n) which can represent dependent features.
    '''
    # -- creating a meshgrid
    X1,X2 = np.meshgrid(np.arange(np.min(X[:,0])-5,np.max(X[:,0])+5,0.5),np.arange(np.min(X[:,1])-5,np.max(X[:,1])+5,0.5))
    Z = multivariateGaussian(np.stack([X1.ravel(),X2.ravel()],axis = 1), mu, Sigma2)
    Z = Z.reshape(X1.shape)

    # -- plotting data points
    plt.plot(X[:,0], X[:,1], 'x', mec = 'b', mew = 2, ms = 8)
    
    # -- do not plot infinities
    if np.all(abs(Z) != np.inf):
        # -- levels = contour levels to show. here we show levels at [1.e-20 1.e-17 1.e-14 1.e-11 1.e-08 1.e-05 1.e-02]
        # -- 1e-20 = definately a outlier --> lowest probability density
        # -- 1e-2 = not a outlier --> highest probability density
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder=100)
    
def selectThreshold(yval, pval):
    '''
    Find the best threshold (epsilon) to use for selecting outliers based
    on the results from a validation set and the ground truth.
    
    : param yval : Find the best threshold (epsilon) to use for selecting outliers based
                   on the results from a validation set and the ground truth.
    : param pval : The ground truth labels of shape (m, ).
    
    : return : bestEpsilon, bestF1
    bestEpsilon : A vector of shape (n,) corresponding to the threshold value.
    bestF1 : The value for the best F1 score.
    '''
    # -- ignoring  RuntimeWarning: invalid value encountered in long_scalars from numpy
    np.seterr(all = 'ignore')
    # -- check if pval and yval are of same shape. If not then make them of same shape.
    if (pval.shape != yval.shape):
        pval = pval.reshape(yval.shape)
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    step = (max(pval) - min(pval))/1000
    for epsilon in np.arange(pval.min(), pval.max(), step):
        # -- prediction is a binary vector. False represents normal values and True represents outliers
        predictions = pval < epsilon
        
        # -- calculating confusion matrix
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))
        
        # -- calculating precision and recall
        prec = (tp)/(tp+fp)
        rec = (tp)/(tp+fn)
        F1 = (2 * prec * rec)/(prec + rec)
        
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb=0.0):
    '''
    Collaborative filtering cost function.
    
    : param params : The parameters which will be optimized. This is a one
                    dimensional vector of shape (num_movies x num_users, 1). It is the 
                    concatenation of the feature vectors X and parameters Theta.
    
    : param Y : A matrix of shape (num_movies x num_users) of user ratings of movies.
    
    : param R : A (num_movies x num_users) matrix, where R[i, j] = 1 if the 
                i-th movie was rated by the j-th user.
                
    : param num_users : int. Total number of users.
    
    : param num_features : int. Number of features to learn.
    
    : param lamb : float, optional. The regularization coefficient.
    
    : returns : J, grad
    J : float. values of cost function at given parameters.
    grad : array of shape (num_movies x num_users,1). The gradient vector of the cost function at given params. 
    '''
    # -- unfold X and theta matrices from params
    # -- X is the movie feature matrix. Shape of (number of movies x number of features)
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    # -- theta is the feature matrix. Shape of (number of users x number of features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    
    # -- initilising return values
    # -- cost function
    J = 0
    # -- derivate of costfunction w.r.t features
    X_grad = np.zeros(X.shape)
    # -- derivative of costfunction w.r.t theta
    Theta_grad = np.zeros(Theta.shape)
    
    # -- cost function with regularization
    J = (0.5) * np.sum(np.square((np.matmul(X, Theta.T) - Y)*R)) + ((lamb/2) * (np.sum(np.square(X)))) + ((lamb/2) * (np.sum(np.square(Theta))))
    
    # -- implementing gradients
    X_grad = np.matmul( (np.matmul(X, Theta.T) - Y) * R , Theta) + (lamb * X)
    Theta_grad = np.matmul( ((np.matmul(X, Theta.T) - Y) * R).T , X) + (lamb * Theta)
    
    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad

def computeNumericalGradient(J, theta, e = 1e-4):
    '''
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.
    
    : param J : func
    The cost function which will be used to estimate its numerical gradient.
    : param theta : The one dimensional unrolled network parameters. The numerical gradient is computed at
                    those given parameters.
    : param e : float (optional)
                The value to use for epsilon for computing the finite difference.
    : returns : numgrad
    : numgrad : array_like
                The numerical gradient with respect to theta. Has same shape as theta.
    '''
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad
        
def checkCostFunction(lamb):
    '''
    Creates a collaborative filtering problem to check your cost function and gradients.
    It will output the  analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result
    in very similar values.
    
    : param lamb : float, optional
        The regularization parameter.
    '''
    # -- random initialisation of feature matrix (X) and parameter matrix (Theta) for calculattion of Y and R.
    # -- creating random matrix of X. 4 movies and 3 features
    X_t = np.random.rand(4, 3)
    # -- creating random matrix of Theta_t. 5 user and 3 features.
    Theta_t = np.random.rand(5, 3)
    
    # -- Movie rating matrix
    Y =  np.matmul(X_t, Theta_t.T)
    # -- randomly making values zero. It means that user didn't rate the movie.
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    # -- creating matrix R. Where Y != 0, R should be 1.
    R = np.zeros(Y.shape)
    R[Y != 0] = 1
    
    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]
    
    params = np.concatenate([X.ravel(), Theta.ravel()])
    # -- calculate numerical gradient using computeNumericalGradient
    numgrad = computeNumericalGradient(lambda x: cofiCostFunc(x, Y, R, num_users, num_movies, num_features, lamb), params)
    # -- gradient from our implementation
    cost, grad = cofiCostFunc(params, Y, R, num_users,num_movies, num_features, lamb)
    
    print(np.stack([numgrad, grad], axis=1))
    print('\nThe above two columns you get should be very similar.'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then '
          'the relative difference will be small (less than 1e-9).')
    print('\nRelative Difference: %g' % diff)
    
def loadMovieList():
    '''
    Reads the fixed movie list in movie_ids.txt and returns a list of movie names.
    
    : returns : movieNames
    : movieNames : list. A list of strings, representing all movie names.
    
    '''
    # Read the fixed movieulary list
    with open('movie_ids.txt', encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movieNames = []
    for movie in movies:
        parts = movie.split()
        # -- skip the index
        movieNames.append(' '.join(parts[1:]).strip())
    return movieNames
    
def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    
    : param Y : The user ratings for all movies. A matrix of shape (num_movies x num_users).
    : param R : Indicator matrix for movies rated by users. A matrix of shape (num_movies x num_users).
    
    : returns : Ynorm, Ymean : array_like
    Ynorm : A matrix of same shape as Y, after mean normalization.
    Ymeans : A vector of shape (num_movies, ) containing the mean rating for each movie.
    """
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean