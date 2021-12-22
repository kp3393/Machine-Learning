'''
This file is in supplement with svm.py. Here we implement all the problems related to part1 explained in the ex6.pdf
NOTE:
1.  Instead of implementing my own SVM, I will be using sklearn's module here for SVM. Since, implementing SVM from
    scratch is not the aim of this exercise.
2.  'kernel = linear' option implements linear SVM.
3.  'kernel = rbf' implements gaussian kernel. Here instead of dividing by sigmasquared,
    it multiplies by 'gamma'. As long as we set gamma = sigma^(-2), it will work just the same.
4.  Does not includes spam classification exercise. The exercise required complete preprocessing of email data which
    is beyond the cope of this exercise and is not ML related.
5.  All the decision boundaries are displayed along with their margin. The data points circled in black are support vectors.
'''

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from supportVM import *

# -- load data from ex6data1
data = read_data('ex6data1.mat')
X = data['X']
y = data['y']

# -- plot data
plotData(X,y,'fig1.png','Example Dataset 1')

# -- training a linear support vector machine for learn about class boundary.
sv_data1 = svm.SVC(C = 1, kernel = 'linear')
# -- we need to flatten y axis because it is shape (m,1) and .fit expects it to be (m,) format
sv_data1_fit = sv_data1.fit(X,y.flatten())
# -- visualize boundary condition
visualizeBoundary(X,y,sv_data1_fit,'fig2.png','SVM maximum margin seperating hyperplane with C = 1')

sv_data1 = svm.SVC(C = 100, kernel = 'linear')
# -- we need to flatten y axis because it is shape (m,1) and .fit expects it to be (m,) format
sv_data1_fit = sv_data1.fit(X,y.flatten())
# -- visualize boundary condition
visualizeBoundary(X,y,sv_data1_fit,'fig3.png','SVM maximum margin seperating hyperplane with C = 100')

# -- checking gaussain implementation
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f is %f' %(sigma, sim))

# -- load data from ex6data2
data = read_data('ex6data2.mat')
X = data['X']
y = data['y']
plotData(X,y,'fig4.png','Example Dataset 2')

# -- training a non linear support vector machine for learn about class boundary.
sigma = 0.1
gamma = np.power(sigma,-2.)
sv_data2 = svm.SVC(C = 1, gamma = gamma,kernel = 'rbf')
# -- we need to flatten y axis because it is shape (m,1) and .fit expects it to be (m,) format
sv_data2_fit = sv_data2.fit(X,y.flatten())
visualizeBoundary(X,y,sv_data2_fit,'fig5.png','SVM maximum margin seperating hyperplane with C = 1 and sigma = 0.1')

# -- load data from ex6data2
data = read_data('ex6data3.mat')
# -- training set
X = data['X']
y = data['y']
# -- validation set
Xval = data['Xval']
yval = data['yval']
plotData(X,y,'fig6.png','Example Dataset 3')

# -- optimized SVM parameters
CVal = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
sigmaVal = CVal
 
C, sigma, best_score = bestParam(CVal,sigmaVal,X,y,Xval,yval)
print("Best C, sigma pair is (%f, %f) with a score of %f." %(C, sigma, best_score))

# -- training a non linear support vector machine from our learned paramters
gamma = np.power(sigma,-2.)
sv_data2 = svm.SVC(C = C, gamma = gamma,kernel = 'rbf')
# -- we need to flatten y axis because it is shape (m,1) and .fit expects it to be (m,) format
sv_data2_fit = sv_data2.fit(X,y.flatten())
visualizeBoundary(X,y,sv_data2_fit,'fig7.png','SVM maximum margin seperating hyperplane with C = %s and sigma = %s'%(str(C),str(sigma)))