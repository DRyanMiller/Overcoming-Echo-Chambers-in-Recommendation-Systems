# -*- coding: utf-8 -*-

# custom.py
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
from joblib import load


def test_custom():
    print("In custom module")
    return None


def cluster_distances(centroids):
    #centroids_array = centroids.to_numpy()
    cluster_distance_matrix = distance_matrix(centroids,
                                              centroids, p=2)
    cluster_distance_df = pd.DataFrame(cluster_distance_matrix)
    return cluster_distance_df


def nearest_clusters(cluster_distance_df, cluster, num_nearest_clusters=2):
    sorted_distances = cluster_distance_df[cluster].sort_values(ascending=True)
    return sorted_distances[1:num_nearest_clusters+1].index.values.astype(int)


def centroid_ratings(centroids, item_factors_unstacked):
    item_factors_unstacked_transposed = item_factors_unstacked.T
    centroid_ratings = np.dot(centroids, item_factors_unstacked_transposed)
    centroid_ratings_df = pd.DataFrame(centroid_ratings)
    centroid_ratings_df.columns = item_factors_unstacked.index
    centroid_ratings_T_df = centroid_ratings_df.transpose()
    return centroid_ratings_T_df


def top_rated_movies(cluster, centroids, item_factors_unstacked):
    centroid_ratings_T_df = centroid_ratings(centroids, item_factors_unstacked)
    sorted_ratings = centroid_ratings_T_df[cluster].sort_values(ascending=False)
    sorted_ratings_df = sorted_ratings.reset_index()
    sorted_ratings_df.columns = ['id', 'rating']
    most_rated = pd.read_csv('../data/processed/most_rated.csv', index_col='Unnamed: 0')
    top_movies = pd.merge(sorted_ratings_df, most_rated, how='inner', left_on='id', right_on='movieId')
    #top_movies.columns = ['id', 'rating', 'movieId', 'title', 'genres']
    #top_100_movies = top_movies.sort_values(by='rating', ascending=False )[:100]
    return top_movies.title


def get_recommendations(user_cluster):
    centroids = pd.read_csv('../data/processed/centroids.csv', index_col=['Unnamed: 0'])
    item_factors_unstacked = pd.read_csv('../data/processed/item_factors_unstacked.csv', index_col=['id'])
    cluster_distance_df = cluster_distances(centroids)
    near_clusters = nearest_clusters(cluster_distance_df, user_cluster)
    recommendation_set = set()
    for index, cluster in enumerate(near_clusters):
        if index==0:
            recs = np.random.choice(top_rated_movies(cluster, centroids, item_factors_unstacked), size=6, replace=False)
            recommendation_set.update(set(recs))
        if index==1:
            cluster_unique_top_movies = set(top_rated_movies(cluster, centroids, item_factors_unstacked)).difference(recommendation_set)
            recs = np.random.choice(list(cluster_unique_top_movies), size=4, replace=False)
            recommendation_set.update(recs)
    return list(recommendation_set)

def user_factors(user_ratings):
    rated_movies = [float(x[0]) for x in user_ratings]
    item_factors_pdf = pd.read_csv('../data/processed/item_factors.csv', index_col='Unnamed: 0')
    rated_item_factor = item_factors_pdf.loc[item_factors_pdf['id'].isin(rated_movies)].pivot(index='id', columns='value', values='features')
    M = rated_item_factor.values
    E = np.identity(42)
    nui = len(rated_movies)
    regParam = 0.15
    R = np.array([float(x[1]) for x in user_ratings])
    A = M.T.dot(M)+regParam*nui*E
    V = M.T.dot(R.T)
    user_fac = np.linalg.inv(A).dot(V)
    return user_fac

def ALS_recommendations(user_fac):
    item_factors_unstacked = pd.read_csv('../data/processed/item_factors_unstacked.csv', index_col=['id'])
    user_movie_ratings = user_fac.dot(item_factors_unstacked.T)
    user_movie_ratings_df = pd.DataFrame(user_movie_ratings)
    user_movie_ratings_df['movieId'] = item_factors_unstacked.T.columns
    user_top_10 = user_movie_ratings_df.sort_values(0,ascending=False).head(10)
    movies_df = pd.read_csv('../data/raw/movies.csv')
    user_top_10 = user_top_10.merge(movies_df, how='left', on='movieId')
    user_top_10.drop([0,'movieId', 'genres'], axis=1, inplace=True)
    return list(user_top_10.title)


def get_user_cluster(user_fac):
    gbc = load('../models/fifp_classification.joblib') 
    user_cluster = gbc.predict(user_fac.reshape(1, -1))[0]
    return user_cluster


def user_rankings():    
    top_100 = pd.read_csv('../data/processed/top_100.csv', index_col='Unnamed: 0')
    ranking_list = top_100.sample(n=50, axis=0)
    user_ratings = []
    i = 0
    j = 0
    while j<9:
        title = ranking_list['title']
        movieId = ranking_list['movieId']
        #print(title.iloc[i])
        user_rating = input('Enter a ranking for {} from 1 (lowest) to 5 (highest). If you have not seen the movie, press enter.'.format(title.iloc[i]))
        i +=1
        j = len(user_ratings)
        if user_rating=='':
            pass
        elif float(user_rating) in range(0,6):
            user_ratings.append((movieId.iloc[i], user_rating))
        else:
            print('Ratings must be whole numbers between 1 and 5.')
            pass
    return user_ratings