import numpy as np
import pandas as pd
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the rating and movie files
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

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

# Find Lowest and Highest rated movies:
mean_rating = ratings.groupby('movieId')[['rating']].mean()

# Lowest rated movies
lowest_rated = mean_rating['rating'].idxmin()
print(movies.loc[movies['movieId'] == lowest_rated])

# Highest rated movies
highest_rated = mean_rating['rating'].idxmax()
print(movies.loc[movies['movieId'] == highest_rated])

# Show number of people who rated the highest and lowest rated movies
print(ratings[ratings['movieId'] == highest_rated])
print(ratings[ratings['movieId'] == lowest_rated])

from scipy.sparse import csc_matrix

def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csc_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]

    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(1, k):  # Start from 1 to skip the movie itself
        n = neighbour[1][0][i] if show_distance else neighbour[0][i]
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

from scipy.sparse.linalg import svds

ut, s, vt = svds(X, k=100)
Q_0 = ut
P_0 = np.transpose(np.diag(s) @ vt)

Y = Q_0 @ np.transpose(P_0)

print('user_0 will rate movie_10 to movie_19 as follows:')
for movie_idx in range(10, 20):
    print(Y[movie_idx, 0])

values = Y[:, 0].tolist()
top_k_indices = np.argsort(values)[-20:][::-1]
print(f"user {user_inv_mapper[0]}, may like the following movies:")
for i in top_k_indices:
    print(movie_inv_mapper[i], movie_titles[movie_inv_mapper[i]])

# 在这里结束程序
sys.exit()

# 使用 SGD 优化矩阵分解
# ... （这部分代码将不会被执行）