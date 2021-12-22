'''
This file is in supplement with km_pca.py. Here we implement all the problems related to PCA explained in the ex7.pdf
'''
import numpy as np
import matplotlib.pyplot as plt
from km_pca import *

data = read_data('ex7data1.mat')
X = data['X']
# routine_2dplot([X[:,0]],[X[:,1]],[''],'x1','x2',['bo'],'Example dataset 1','fig4.png')

# -- feature normalization
X_norm, mu, _ = featureNormalize(X)
U, S = pca(X_norm)

# -- Draw the eigenvectors centered at mean of data.
plt.plot(X[:, 0], X[:, 1], 'bo')
for i in range(2):
    plt.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],head_width=0.25, head_length=0.2, fc='k', ec='k')
plt.title('Computed eigenvector of the dataset')
plt.savefig('fig5.png',format = 'png', dpi = 600, bbox_inches = 'tight')

# -- printing eigen vector
print('Top eigenvector: U[:, 0] = ',U[0, 0], U[1, 0])

# Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K)
print('Projection of the first example: ', Z[0,0],'\n')

# recovery of data
X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: ', X_rec[0,0], X_rec[0,1],'\n');

# -- plotting original and reconstructed.
# -- plotting normalized data points
plt.plot(X_norm[:,0],X_norm[:,1],'bo', label = 'original data point')
# -- plotting reconstructed data points
plt.plot(X_rec[:,0],X_rec[:,1],'ro', label = 'reconstructed data point')
# -- plotting projection lines
for xnorm,xrec in zip(X_norm,X_rec):
    plt.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
plt.legend()
plt.title('Normalized and projected data after PCA.')
plt.savefig('fig6.png',format = 'png', dpi = 600, bbox_inches = 'tight')