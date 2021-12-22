'''
This file is in supplement with ad_rs.py. Here we implement all the problems related to recommender system explained in the ex8.pdf
'''
import numpy as np
import matplotlib.pyplot as plt
from ad_rs import *

# -- READING INPUT DATA
data = read_data('ex8_movies.mat')
# -- Y is a 1682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
Y = data['Y']
# -- R is a 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
R = data['R']
print('Average rating for movie 1 (Toy Story): %f / 5' %np.mean(Y[0, R[0, :]]),'\n')

# -- We can "visualize" the ratings matrix by plotting it with imshow
# plt.figure(figsize=(8, 8))
# plt.imshow(Y)
# plt.ylabel('Movies')
# plt.xlabel('Users')
# plt.savefig('fig4.png',format = 'png', dpi = 600, bbox_inches = 'tight')

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = read_data('ex8_movieParams.mat')
# --  assigning paramters
X, Theta, num_users, num_movies, num_features = data['X'], data['Theta'], data['num_users'], data['num_movies'], data['num_features']

# -- testing our implementation of cost function calculation
#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, 0:num_users]
R = R[:num_movies, 0:num_users]

#  Evaluate cost function unregularized
J, grad = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]), Y, R, num_users, num_movies, num_features)
print('Cost at loaded parameters: ', J, '\n')

# -- checking implementation of unregularized gradients
# checkCostFunction(0)

#  Evaluate cost function regularized
J, grad = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]), Y, R, num_users, num_movies, num_features, 1.5)
print('Cost at loaded parameters (lambda = 1.5): %.2f' % J)
print('              (this value should be about 31.34)')

# -- checking implementation of regularized gradients
checkCostFunction(1.5)