'''
This file is in supplement with ad_rs.py. Here we implement all the problems related to anomaly detection explained in the ex8.pdf
'''
import numpy as np
import matplotlib.pyplot as plt
from ad_rs import *

data = read_data('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
routine_2dplot([X[:,0]],[X[:,1]],[''],'Latency (ms)','Throughput (mb/s)','x','The first dataset','fig1.png')

# -- rechecking gaussian implementation from test case provided

a = np.array([[16,2,3,13],[5,11,10,8],[9,7,6,12],[4,14,15,1]])
a = np.sin(a)
a = a[:,:3]
mu,sigma2 = estimateGaussian(a)
print('Obtained values of mu : ',mu,'\n')
print('Expected values of mu are : -0.3978779  0.3892253  -0.0080072','\n')

print('Obtained values of sigma are : ',sigma2,'\n')
print('Expected values of sigma are :  0.27795  0.65844  0.20414','\n')

# -- Estimate mu and sigma for each feature in X
mu, sigma2 = estimateGaussian(X)

# -- return the density of the multivariate normal at each data point in X
p = multivariateGaussian(X, mu, sigma2)
visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Contour plot')
plt.savefig('fig2.png',format = 'png', dpi = 600, bbox_inches = 'tight')

# -- selecting threshold
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: %f' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

# -- Find the outliers in the training set and plot the
outliers = p < epsilon

#  -- Visualize the fit
visualizeFit(X,  mu, sigma2)
# -- Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, mfc='None', mew=2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.title('Classified anomalies')
plt.savefig('fig3.png',format = 'png', dpi = 600, bbox_inches = 'tight')

# -- high dimensional dataset
# -- read the data
data = read_data('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

# -- Estimate mu and sigma for each feature
mu, sigma2 = estimateGaussian(X)

# -- return the density of the multivariate normal at each data point in X
p = multivariateGaussian(X, mu, sigma2)

# -- cross validation set
pval = multivariateGaussian(Xval, mu, sigma2)

# -- find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: %f' % epsilon)
print('Best F1 on Cross Validation Set          : %f\n' % F1)
print('  (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of      0.615385)')
print('\n# Outliers found: %d' % np.sum(p < epsilon))