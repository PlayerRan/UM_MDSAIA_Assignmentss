# movie recommendation using CF algorithm
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

## load the rating file and movie file
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
ratings.head()

movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
movies.head()

n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings / n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings / n_movies, 2)}")

user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
user_freq.head()

# Find Lowest and Highest rated movies:
mean_rating = ratings.groupby('movieId')[['rating']].mean()  # calculate the mean rating for each movieId

# Lowest rated movies
lowest_rated = mean_rating['rating'].idxmin()  # return the idx of the lowest ratings
movies.loc[movies['movieId'] == lowest_rated]
print(movies.loc[movies['movieId'] == lowest_rated])  # print the movies of the lowest_rated

# Highest rated movies
highest_rated = mean_rating['rating'].idxmax()
movies.loc[movies['movieId'] == highest_rated]
print(movies.loc[movies['movieId'] == highest_rated])

# show number of people who rated movies rated movie highest
ratings[ratings['movieId'] == highest_rated]
# show number of people who rated movies rated movie lowest
ratings[ratings['movieId'] == lowest_rated]


# Now, we create user-item matrix using scipy csc matrix
#from scipy.sparse import csc_matrix. the package requires Visual Studio installed in your computer

from scipy.sparse import csc_matrix

# define a function to create the utility matrix of movie X user
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    # Map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    # Map indices to IDs
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]


    X = csc_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))


    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# create the csc matrix, user_mapper, movie_mapper,...
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)


from sklearn.neighbors import NearestNeighbors

"""
Find similar movies using KNN
"""


def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]  #map movie_id to movie_idex
    movie_vec = X[movie_ind]

    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 3

similar_ids = find_similar_movies(movie_id, X, k=20)
movie_title = movie_titles[movie_id]

print(f"Since you watched {movie_title}, you may like the following:")
for i in similar_ids:
    print(movie_titles[i])


## regarding unkown value as 0, solve SVD for X
#ut, s, vt = svds(X, 100) # do SVD, asking for 100 factors.


from scipy.sparse.linalg import svds
ut, s, vt = svds(X,k=100)


#Q_0 movie-feature matrix as in the ppt, can be regarded as initial value of Q
Q_0 = ut
P_0 = np.transpose(np.diag(s) @ vt)  # user-feature matrix as in ppt, can be regarded as initial value of P

Y = Q_0 @ np.transpose(P_0)

print('user_0 will rate movie_10 to movie_19 as follows:')
for movie_idx in range(10,20):  # predict how user_0 rate movie_10 to movie_19, this is a wrong result,
    print(Y[movie_idx,0])       # because now Y is the SVD approximation of X with unknown rating as 0

## predict what movies of the first user likes
values = Y[:,0].tolist()   # get the first column, the scores rated by use_0
top_k_indices = np.argsort(values)[-20:][::-1]   # get the indices of the top 20
print(f"user {movie_inv_mapper[0]}, may like the following movies:")
for i in top_k_indices:
    print(movie_inv_mapper[i], movie_titles[movie_inv_mapper[i]])


## you need to use latent factor decompostion algorithm for seeking for Q,P
## following the PPT, you may seek for Q, P using SGD algorithm, taking Q_0, P_0 as the initial value
## once you solved the problem, please print how user_0 rate movie_10 to movie_19
