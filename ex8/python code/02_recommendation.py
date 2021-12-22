'''
This file is in supplement with ad_rs.py. Here we implement learnings from recommendations.
'''
import numpy as np
from scipy import optimize
from ad_rs import *

movieList = loadMovieList()
n_m = len(movieList)

# -- initialize my ratings
my_ratings = np.zeros(n_m)

# -- Check the file movie_idx.txt for id of each movie in our dataset
# -- For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
# -- remember in python indexing starts from 0. 
# -- This is similar to an user opening account for first time and rating a few movies.
my_ratings[0] = 4

# -- Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2 

# -- We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[52] = 4
my_ratings[62]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\n\nNew user ratings:\n')
print('-----------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d stars: %s' % (my_ratings[i], movieList[i]))

# -- training the collaborative filtering model on a movie rating dataset of 1682 movies and 943 users

# -- Load data
data = read_data('ex8_movies.mat')
Y, R = data['Y'], data['R']

# --  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
# --  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

# --  Add our own ratings to the data matrix
Y = np.hstack([my_ratings[:, None], Y])
R = np.hstack([(my_ratings > 0)[:, None], R])

# --  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

# --  Useful Values
num_movies, num_users = Y.shape
num_features = 10

# -- Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate([X.ravel(), Theta.ravel()])

# -- Set options for scipy.optimize.minimize
options = {'maxiter': 100}

# -- Set Regularization
lambda_ = 10
res = optimize.minimize(lambda x: cofiCostFunc(x, Ynorm, R, num_users, num_movies, num_features, lambda_), initial_parameters, method='TNC', jac=True, options=options)
theta = res.x

# Unfold the returned theta back into U and W
X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)

print('Recommender system learning completed.')

# -- prediction matrix
p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean

movieList = loadMovieList()

ix = np.argsort(my_predictions)[::-1]

print('Top recommendations for you:')
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList[i]))