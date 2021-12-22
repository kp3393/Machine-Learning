'''
This file is in supplement with km_pca.py. Here we implement optional exercise of face image dataset.
'''
import numpy as np
import matplotlib.pyplot as plt
from km_pca import *
# -- Load Face dataset
data = read_data('ex7faces.mat')
X = data['X']
# -- Display the first 100 faces in the dataset
displayData(X[0:100, :], 'fig7.png','Face dataset', example_width = None, figsize=(8, 8))

# -- feature normalise
X_norm, mu, _ = featureNormalize(X)
print('Original dataset X has a shape of : ',X_norm.shape)

# -- run PCA
U, S = pca(X_norm)

# -- diplaying top 36 vectors which are formed
# -- remember to take transpose of U. Ured = nxk
displayData(U[:,0:36].T, 'fig8.png','Principal component on the face dataset', example_width = None, figsize=(8, 8))

# -- dimensionality reduction (K = 100)
K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a shape of: ',Z.shape)

# -- recovered values
K = 100
X_rec = recoverData(Z, U, K)
print('Done')
displayData(X_rec[0:100,:], 'fig9.png', 'Images reconstructed from top 100 principal components',example_width = None, figsize=(8, 8))
