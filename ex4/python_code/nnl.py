'''
This file implements the ex4 (Neural Networks Learning) of Andrew Ng's Machine learning course using Python.
1. reads the data from .mat file
2. sigmoid function 
3. nnCostFunction to calculate regularized and unregularized cost function
4. randInitializeWeights randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
5. debugInitializeWeights Initialize the weights of a layer with fan_in incoming connections and fan_out outgoing connections using a fixed
   strategy, this will help you later in debugging.
6. computeNumericalGradient computes the difference using finite differences
7. sigmoidGradient Computes gradient of a sigmoid function evaluated at z
8. checkNNGradients Creates a small NN to check the BACKPROPOGATION gradients.
9. predict Predict the label of an input, given a trained neural network.

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
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
    
def sigmoid(z):
    '''
    Computes sigmoid of z. g(z) = 1/(1+exp(-z))
    
    :param z : Vector, array or matrices. Input values for which sigmoid function needs to be calculated.
    '''
    g = np.zeros((np.shape(z)))
    g = np.dot(1,(1+np.exp(-z))**-1)
    
    return g

def sigmoidGradient(z):
    '''
    Computes gradient of a sigmoid function evaluated at z.
    :param z : Matrix or a vector. Values for which the gradient has to be calculated.
    
    returns :  g
    g : gradient of the sigmoid function
    '''
    g = np.multiply(sigmoid(z),(1-sigmoid(z)))
    
    return g

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    '''
    Implements the neural network cost function of a TWO layer neural network which performs classification.
    The parameters of the neural network are 'unrolled' into vectors
    
    :param nn_params : Vector. UNROLLED parameters of the neural network
    :param input_layer_size : integer. Feature size of the input layer WITHOUT bias unit
    :param hidden_layer_size : integer. size of the hidden layer without any bias unit
    :param num_labels : integer. number of classification units
    :param X : array of shape(sample size, number of parameters). features of the given dataset. WITHOUT BIAS UNIT
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param lamb : integer. regularization parameter.
    
    returns : J, grad
    J : integer. consisting of Regularied cost function for each set of theta value.
    grad : "unrolled" vector of the partial derivatives of the neural network.
    '''
    # number of samples
    m = X.shape[0]
    # number of features
    n = X.shape[1]
    
    # reshaping nn_param back into parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    # things we want to return
    Theta1_grad = np.zeros((Theta1.shape))
    Theta2_grad = np.zeros((Theta2.shape))
    
    # -------------------- FEED FORWARD & Cost Function Regularized and Unregularized -------------------- #
    # input layer 1. adding bias unit as well. (5000x401)
    a1 = np.append((np.ones((m,1))),X,1)
    # making it 401x5000
    a1 = np.transpose(a1)
    
    # hidden layer. Adding bias unit as well. (26x5000)
    a2 = np.append((np.ones((1,m))),sigmoid(np.matmul(Theta1,a1)),0)
    
    # output layer. Hypothesis vector. (10x5000)
    a3 = sigmoid(np.matmul(Theta2,a2))
    
    # output array is mx1. we need to make it num_labels x m.
    y_k = np.zeros((num_labels,m))
    
    # remember that the indexing in python starts from 0. 0th index corresponds to 1, 1st index corresponds to 2...9th index corresponds to 10
    for row in range(m):
        # suppose y = 5, i.e. the 4th index of y_k should be one. Thus, y[row]-1 row of the mth column will be one.
        y_k[y[row]-1,row] = 1
    
    # calculating unregularized cost function
    J = (-1/m)*sum(sum(np.multiply(y_k,np.log(a3))+np.multiply((1-y_k),(np.log(1-a3)))))
    
    # calculating regularized cost function
    J = (-1/m)*sum(sum(np.multiply(y_k,np.log(a3))+np.multiply((1-y_k),(np.log(1-a3))))) + ((lamb/(2*m))*(sum((sum(Theta1[:,1:]**2)))+sum(sum(Theta2[:,1:]**2))))
    
    # -------------------- BACKPROPOGATION for Gradient Calculation -------------------- #
    # error for output layer
    d3 = a3 - y_k
    
    # error for hidden layer
    d2 = np.multiply(np.matmul(np.transpose(Theta2),d3)[1:,:],sigmoidGradient(np.matmul(Theta1,a1)))
        
    # Theta2_grad
    Theta2_grad = (1/m)*(np.matmul(d3,np.transpose(a2)))
    
    # Theta1_grad
    Theta1_grad = (1/m)*(np.matmul(d2,np.transpose(a1)))
    
    # for regularized NN
    # print(Theta1,'\n')
    # print(np.append(np.zeros((Theta1.shape[0],1)),Theta1[:,1:],1),'\n')
    
    Theta1_reg = (lamb/m)*(np.append(np.zeros((Theta1.shape[0],1)),Theta1[:,1:],1))
    Theta2_reg = (lamb/m)*(np.append(np.zeros((Theta2.shape[0],1)),Theta2[:,1:],1))
    
    Theta1_grad = Theta1_grad + Theta1_reg
    Theta2_grad = Theta2_grad+Theta2_reg
    
    grad = np.append(Theta1_grad.flatten(),Theta2_grad.flatten())
    return J,grad
    
def randInitializeWeights(L_in,L_out):
    '''
    Randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
    
    :param L_in : number of incoming connections
    :param L_out : number of outgoing connections
    
    returns : W
    W : matrix of shape(L_out,1 + L_in)
    '''
    epsilon_init = 0.12
    W = np.zeros((L_out, 1+L_in))
    W = (np.random.rand(L_out,1+L_in))*((2*epsilon_init)-(epsilon_init))
    
    return W

def debugInitializeWeights(fan_out,fan_in):
    '''
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging.
    
    :param fan_out : outgoing connections
    :param fan_in : incoming connections
    
    returns : W
    W : matrix of shape (fan_out,1 + fan_in). Weights for each layer.
    '''
    W = np.zeros((fan_out,fan_in+1))
    W = np.reshape(np.sin(np.arange(1,W.size + 1)),W.shape)/10
    return W
    
def computeNumericalGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lamb):
    '''
    Computes the gradient using "finite differences" and gives us a numerical estimate of the gradient.
    
    :param nn_params : Vector. UNROLLED parameters of the neural network
    :param input_layer_size : integer. Feature size of the input layer WITHOUT bias unit
    :param hidden_layer_size : integer. size of the hidden layer without any bias unit
    :param num_labels : integer. number of classification units
    :param X : array of shape(sample size, number of parameters). features of the given dataset. WITHOUT BIAS UNIT
    :param y : array of shape(sample size, 1).outputs of the given dataset
    :param lamb : integer. regularization parameter.
    
    returns : numgrad
    numgrad : vector of shape (nn_params,1). Partial derivative of J wrt thetas
    '''
    numgrad = np.zeros((nn_params.shape))
    perturb = np.zeros((nn_params.shape))
    epi = 1e-4
    for row in range(nn_params.size):
        perturb[row] = epi
        loss1 = nn_params - perturb
        J_left,_ = nnCostFunction(loss1, input_layer_size, hidden_layer_size,num_labels, X, y, lamb)
        loss2 = nn_params + perturb
        J_right,_ = nnCostFunction(loss2, input_layer_size, hidden_layer_size,num_labels, X, y, lamb)
        
        numgrad[row] = (J_right-J_left)/(2*epi)
        perturb[row] = 0
    
    return numgrad
def checkNNGradients(lamb):
    '''
    Creates a small NN to check the BACKPROPOGATION gradients.
    It will output the analytical gradients produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should result in very similar values.
    
    :param lamb : integer. regularization parameter
    
    returns : prints the analytical gradients produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient), also the relative difference between both.
    '''
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    # we need to generate some random test data for our neural network.
    Theta1 = debugInitializeWeights(hidden_layer_size,input_layer_size)
    Theta2 = debugInitializeWeights(num_labels,hidden_layer_size)
    # generating input X
    X = debugInitializeWeights(m,input_layer_size-1)
    y = 1 + np.transpose(np.arange(1,m+1)%num_labels)
    y = y.reshape(m,1)
    
    # unroll the parameters
    nn_params = np.append(Theta1.flatten(),Theta2.flatten())
    
    # calculating the costfunction and gradient using backpropogation
    cost,grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lamb)
    # calculating gradient using computeNumericalGradient
    numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size,num_labels, X, y, lamb)
    
    # Visually examine the two gradient computations.  The two columns you get should be very similar. 
    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')
    
    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)
    
    print('If your backpropagation implementation is correct, then \n'
    'the relative difference will be small (less than 1e-9). \n'
    'Relative Difference: %g' % diff)

def predict(Theta1,Theta2,X):
    '''
    Predict the label of an input, given a trained neural network.
    
    :param Theta1 : array of shape(hidden_layer_size,features+1). Parameters from layer 1 to layer 2.
    :param Theta2 : array of shape(num_labels,hidden_layer_size+1). Parameters from Layer 2 to layer 3.
    
    returns: p 
    p : array of shape (sample size,1). outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2) 
    '''
    # number of samples
    m = X.shape[0]
    # number of features
    n = X.shape[1]
    a1 = np.append((np.ones((m,1))),X,1)
    # making it 401x5000
    a1 = np.transpose(a1)
    
    # hidden layer. Adding bias unit as well. (26x5000)
    a2 = np.append((np.ones((1,m))),sigmoid(np.matmul(Theta1,a1)),0)
    
    # output layer. Hypothesis vector. (10x5000)
    a3 = sigmoid(np.matmul(Theta2,a2))
    # find the index with max probability. add one to index because the indexing starts from 0.
    p = np.argmax(a3, axis=0) + 1
    
    # do not forget to reshape. We need it in mx1 form for comparison.
    return p.reshape(m,1)
