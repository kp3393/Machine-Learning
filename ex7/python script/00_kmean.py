'''
This file is in supplement with km_pca.py. Here we implement all the problems related to K mean explained in the ex7.pdf
'''
import numpy as np
import matplotlib.pyplot as plt
from km_pca import *

# -- Load a dataset which we will be using
data = read_data('ex7data2.mat')
X = data['X']

# -- selecting initial set of centroids
# -- 3 centroids
K = 3
initial_centroid = np.array([[3,3],[6,2],[8,5]])

# -- find the closest centroids for the examples using the initial_centroid
idx = findClosestCentroids(X,initial_centroid)
print('Closest centroids for the first 3 examples: ','\n',idx[:3])

# -- Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids:','\n',centroids)

# Run K-Means algorithm. The 'true' at the end tells our function to plot the 
centroids, idx = runkMeans(X, initial_centroid, max_iters=10, plot_progress=True)

# -- Image compression using kmean
img = imageRead('bird_small.png')
# -- divide by 255 so that each value is between 0 - 1
img /= 255
# -- reshaping image in Nx3 where N is number of pixels
M,N,P = img.shape
X = np.reshape(img,(M*N,P))

# -- running K mean algorithm on this data
# -- K is 16 because we want to represent each pixel from these 16 colors
K = 16
max_iters = 10

# -- randomly initialising centroids
initial_centroids = kMeanInitCentroids(X,K)
centroids, idx = runkMeans(X, initial_centroids, max_iters=max_iters)
# -- assigning corresponding centroid to each pixel in X
X_recovered = centroids[idx,:]
# -- reshaping X_recovered into M,N,P to display the compressed image back
X_recovered = X_recovered.reshape((M,N,P))

# -- plotting image
# Display the original image, rescale back by 255
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img*255)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed, with %d colors' % K)
ax[1].grid(False)

plt.savefig('bird_small_comp.png',format = 'png', dpi = 600, bbox_inches = 'tight')
