'''
This python script implements exerice 7 of Andrew Ng's Machine learning course. Problem is explained in the pdf ex7.pdf.
Functions implemented
1. read_data
2. findClosestCentroids
3. computeCentroids
4. imageRead
5. routine_2dplot
6. featureNormalize
7. pca
8. projectData
9. recoverData
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
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

def findClosestCentroids(X, centroids):
    '''
    Computes the centroid membership for every example.
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param centroids : array of shape (K,n). K-mean centroid cluster with 'K' centroids and 'n' dimensions.
    
    :returns : idx
    idx : Vector of shape (m,) which hold the centroid assignment for each example.
    '''
    # -- initialising m, n and k as help variables
    m,n = X.shape
    K = centroids.shape[0]
    # -- vector to be returned
    idx = np.zeros((m,1),dtype = np.int)
    
    # -- looping over each example to find the nearest centroid.
    for example in range(m):
        # -- assuming best minimum distance between the point 
        # -- and the associated centroid to be infinity.
        best_min_dist = float('inf')
        for centroid in range(K):
            # -- for each centroid calculate the distance between the example
            this_min_distance = np.sum((X[example,:]-centroids[centroid,:])**2)
            # -- swap the values if this_min_distance < best_min_dist
            if this_min_distance < best_min_dist:
                best_min_dist = this_min_distance
                idx[example,0] = centroid
    
    return idx

def computeCentroids(X, idx, K):
    '''
    Computes new centroid by computing the mean of the data points assigned to each centroid.
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param idx : A vector of size (m,1) of centroid assignments (i.e. each entry in range [0 ... K-1]).
    :param K : int. Number of clusters.
    
    :returns : centroids
    centroids : array of shape (K,n) where each row is the mean of the data points assigned to it.
    '''
    # -- initialising m, n as help variables
    m,n = X.shape
    
    # -- vector to be returned
    centroids = np.zeros((K,n))
    
    for k in range(K):
        c_k = np.where(idx == k)[0]
        centroids[k,:] = (1/c_k.size)*(sum(X[c_k,:]))
    return centroids 

def plotProgresskMeans(i, X, centroid_history, idx_history):
    '''
    A helper function that displays the progress of k-Means as it is running. It is intended for use
    only with 2D data. It plots data points with colors assigned to each centroid. With the
    previous centroids, it also plots a line between the previous locations and current locations
    of the centroids.
    
    :param i : int. Current iteration number of k-means. Depends on number of frames.
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param centroid_history : list. A list of computed centroids for all iteration.
    : param idx_history : list. A list of computed centroids for all iteration.
    '''
    # -- finding number of centroids
    K = centroid_history[0].shape[0]
    # -- Get Current Figure and clear all previous layouts on it
    plt.gcf().clf()
    cmap = plt.cm.rainbow
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    
    for k in range(K):
        # -- need to get position of each centroid after each iteration in a stack.
        # -- centroid_history is a list of arrays. each array has K rows and n columns.
        # -- we need to stack each k row of each iteration as they represent update of kth centroid after ith iteration.
        # -- c is the pointer to the array in list. Thus can be addresed using numpy conventions.
        current = np.stack([c[k,:] for c in centroid_history[:i+1]],axis = 0)
        
        # -- plotting centroid
        plt.plot(current[:, 0], current[:, 1],'-Xk',mec='k',lw=2,ms=10,mfc=cmap(norm(k)),mew=2)
        
        # -- plotting scatter plot
        plt.scatter(X[:, 0], X[:, 1],c=idx_history[i][:,0],cmap=cmap,marker='o',s=8**2,linewidths=1,)
        
        plt.grid(False)
        plt.title('Iteration number %d' % (i+1))
    

def runkMeans(X, centroids, max_iters=10, plot_progress=False):
    '''
    Runs the K-means algorithm
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param centroids : array of shape (K,n). K-mean centroid cluster with 'K' centroids and 'n' dimensions.
    :param max_iters : int,optional. Specifies the total number of times the K-mean algorithm is to be implemented. Default to 10.
    :param plot_progress : bool,optional. A flag which indicates if the function should also plot its progress as the learning happens.
                           if True saves the animation as 'anim.mp4' in the working directory.
    
    :returns : centroids, idx, figure
    centroids : array of shape (K,n). Centroid matrix updated after max_iters.
    idx : a vector of shape (m,1) of centroid assignments (i.e. each entry in range [0 ... K-1]) updated after max_iters.
    figure : if plot_progress is true it will save the animation as 'anim.mp4' in the working directory.
    '''
    # -- find ou number of centroids
    K = centroids.shape[0]
    idx = None
    # saves value of idx for each iteration
    idx_history = []
    # saves value of centroids for each iteration
    centroid_history = []
    
    for iters in range(max_iters):
        # --  find which is the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        # -- if plot progress is true than save the idx and centroids for plotting
        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)
        
        # -- update centroids according to assigned examples
        centroids = computeCentroids(X, idx, K)
    if plot_progress:
        # -- initialize a figure object
        fig = plt.figure()
        # arguments of FuncAnimation
        # fig : The figure object that is used to get draw, resize, and any other needed events.
        # plotProgresskMeans : The function to call at each frame.
        # frames : Source of data to pass func and each frame of the animation
        # interval : Delay between frames in milliseconds
        # repeat dealy : If the animation in repeated, adds a delay in milliseconds before repeating the animation
        # fargs : Additional arguments to pass to each call to func.
        anim = animation.FuncAnimation(fig, plotProgresskMeans,frames=max_iters,interval=500,repeat_delay=2,fargs=(X, centroid_history, idx_history))
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('anim.mp4', writer=writer)
    return centroids, idx

def kMeanInitCentroids(X, K):
    '''
    This function initializes K centroid that are to be used in K-means on the dataset x.
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param K : int. The number of clusters.
    
    :returns : centroids
    centroids : randomly selected K datapoints as centroids. This is a matrix of shape (K,n).
    '''
    m,n = X.shape
    
    # -- values to be returned
    centroids = np.zeros((K,n))
    
    # -- random initialisation
    # -- randomly reorder the indices of examples
    randidx = np.random.permutation(m)
    # -- take first K examples as centroids
    centroids = X[randidx[:K],:]
    
    return centroids
    
def imageRead(fname):
    '''
    Reads image from the current directory.
    
    :param fname : image which has to be read.
    
    returns : img
    img : The image data. The returned array has shape
    (M, N) for grayscale images.
    (M, N, 3) for RGB images.
    (M, N, 4) for RGBA images.
    '''
    img = mpl.image.imread(fname)
    return img
    
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

def pca(X):
    '''
    Run principal component analysis.
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    
    :returns: U, S
    U : array of shape (n x n)
        The eigenvectors, representing the computed principal components
        of X. Each column is a single principal component.
    S : vector of size n, contaning the singular values for each
        principal component. Note this is the diagonal of the matrix we 
        mentioned in class.
    '''
    # -- initialising m, n as help variables
    m,n = X.shape
    sigma = (1/m)*(np.matmul((X.T),X))
    U, S, V = np.linalg.svd(sigma)
    
    return U, S
    
def projectData(X, U, K):
    '''
    Computes the reduced data represntation when projecting only on to the top K Eigenvectors.
    
    :param X : array of shape (m,n). Input dataset with 'm' examples each with 'n' dimensions.
    :param U : array of shape (n,n). The computed eigenvectors using PCA. Each column in the matrix represents a single
               eigenvector (or a single principal component).
    :param K : int. Number of dimensions to project onto. Must be smaller than n.
    
    :returns : Z
    Z : array of shape(m,k). The projects of the dataset onto the top K eigenvectors. 
    '''
    Z = np.matmul(X,U[:,:K])
    return Z
    
def recoverData(Z, U, K):
    '''
    Recovers an approximation of the original data when using the 
    projected data.
    
    Parameters
    ----------
    Z : array of shape (m,k). The reduced data after applying PCA.
    
    U : array of shape (n,n).
        The eigenvectors (principal components) computed by PCA. Each column represents
        a single eigenvector.
    
    K : int
        The number of principal components retained
        (should be less than n).
    
    : returns : X_rec
    X_rec : array of shape (m,n)
            The recovered data after transformation back to the original 
            dataset space.
    '''
    X_rec = np.matmul(Z,U[:,0:K].T)
    return X_rec

def displayData(X, fname, title, example_width = None, figsize = (10,10)):
    '''
    Displays 2D data in a nice grid.
    
    :param X : The input data of size (m x n) where m is the number of examples and n is the number of
               features.
    :param fname : string. save image with this name as a png file in the present directory with a dpi = 600.
    :param title : string for graph title
    :param example_width : int, optional. The width of each 2-D image in pixels. If not provided, the image is assumed to be square,
                            and the width is the floor of the square root of total number of pixels.
    :param figsize : tuple, optional. A 2-element tuple indicating the width and height of figure in inches.
    
    :output : saves the image in the working directory with name fname.png
    '''
    # -- compute number of rows and cols for a given X
    if X.ndim == 2:
        m,n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        # -- make it a 2D array
        X = X.reshape((m,n))
    else : 
        raise IndexError('Input X should be 1 or 2 dimensional.')
    
    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n/example_width)
    
    # -- how many items to be displayed
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    
    # -- displaying images
    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)
    
    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    
    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')
    
    # -- save image
    plt.gcf().suptitle(title)
    plt.savefig(fname,format = 'png', dpi = 600, bbox_inches = 'tight')