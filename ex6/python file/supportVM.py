'''
This python script implements exerice 6 of Andrew Ng's Machine learning course. Problem is explained in the pdf ex6.pdf.
Functions implemented
1. read_data
2. routine_2dplot
3. plotData
4. visualizeBoundary
5. gaussianKernel
6. bestParam

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

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
    
def plotData(X,y,fname,title):
    '''
    Plots the data points X and y into a new figure
    Data points with + for the positive examples and o for the negative examples
    X is assumed to be a Mx2 matrix
    
    :param X : array of shape (M,2) where M is number of samples. Input samples with their features
    :param y : array of shape (M,1). Labels for input arrays
    :param fname : saves the image in the working directory with the name fname.png
    :param title : string. title of the image.
    
    output:
    a graph/image is saved in the working directory with the given image name.
    '''
    # -- find positive and negative labels in pos and neg
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    
    routine_2dplot([X[pos,0], X[neg,0]],[X[pos,1], X[neg,1]],['pos','neg'],'x1','x2',['X','o'],title,fname)
    
def visualizeBoundary(X,y,model,imgname,title):
    '''
    Plots a decision boundary with margin learned by the SVM.
    
    :param X : array of shape (m x 2). The training data with two features (to plot in a 2-D plane).
    :param y : array of shape (m,1). Data labels
    :param model : Trained svm model.
    :param imgname : string. saves the image in the working directory with the given file name.
    :param title : string. title of the image.
    
    output:
    a graph/image is saved in the working directory with the given image name.
    All the decision boundaries are displayed along with their margin. 
    The data points circled in black are support vectors.
    '''
    # -- STEP 1 : Displaying the data points
    # -- find positive and negative labels in pos and neg
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.scatter(X[pos,0],X[pos,1],marker = 'X')
    plt.scatter(X[neg,0],X[neg,1],marker = 'o')
    
    # -- STEP 2 : creating grid to evaluate model
    # -- creating a 1D array
    h = 0.1
    xx = np.linspace(X[:,0].min()-h,X[:,0].max()+h)
    # -- creating a 1D array
    yy = np.linspace(X[:,1].min()-h,X[:,1].max()+h)
    # -- meshgrid for each point in the xy plane
    XX,YY = np.meshgrid(xx,yy)
    # -- creates a each x with each y value
    xy = np.c_[XX.ravel(),YY.ravel()]
    # -- evaluate the model at these values of xy
    Z = model.decision_function(xy).reshape(XX.shape)
    
    # -- STEP 3 : plotting decision boundary and margins
    # -- levels are -1,0,1 bcoz at 0 we have our hyperplane. -1 and +1 are our constrains of decision boundary. 
    # -- anything below -1 is -ve and above +1 is postive
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],linestyles=['--', '-', '--'])
    
    # -- STEP 4 : plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none', edgecolors='k')
    
    # -- STEP 5 : save fig
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.xlim(XX.min(),XX.max())
    plt.ylim(YY.min(),YY.max())
    plt.savefig(imgname,format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.clf()
    
def gaussianKernel(x1,x2,sigma):
    '''
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.
    
    :param x1 : a vector of size (n, ), representing the first datapoint.
    :param x2 : a vector of size (n, ), representing the second datapoint.
    :param sigma : float. The bandwidth parameter for the Gaussian kernel.
    
    returns : sim
    sim : float. The computed RBF between the two provided data points.
    '''
    sim = np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))
    return sim
    
def bestParam(CVal,sigmaVal,X,y,Xval,yval):
    '''
    Computes best possible combination of regularization parameter C and sigma.
    
    :param Cval : array (nx1). values of C which needs to be tried
    :param sigmaVal : array (nx1). values of sigma which needs to be tried
    :param X : array of shape (m x n). Matrix of training data where m is number of training examples, 
               and n is the number of features.
    :param y : array of shape (m, ) vector of labels for ther training data.
    :param Xval : array of shape (mv x n) matrix of validation data where mv is the number of validation examples 
                  and n is the number of features
    :param yval : array of shape (mv, ) vector of labels for the validation data.
    
    :returns : C, sigma
    :C : float. Best performing parameter of regularization paramter C.
    :sigma : float. Best performing paramter of RBF paramter sigma.
    :score : float. Score at most optimal performing paramters
    '''
    # # -- assigning a default value for accuracy
    best_score = sigma = C = 0
    
    for c in CVal:
        for s in sigmaVal:
            # -- calculate gamma for rbf
            gamma = np.power(s,-2.)
    
            # -- train the model
            sv_data = svm.SVC(C = c, gamma = gamma, kernel = 'rbf')
            sv_data.fit(X,y.flatten())
    
            # -- Return the mean accuracy on the given test data and labels
            present_score = sv_data.score(Xval,yval)
            if present_score > best_score:
                best_score = present_score
                sigma = s
                C = c
    return C,sigma,best_score