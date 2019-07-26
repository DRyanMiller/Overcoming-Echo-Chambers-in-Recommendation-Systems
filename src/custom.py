# -*- coding: utf-8 -*-

# custom.py
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np


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


def get_recommendations(user_cluster, centroids, item_factors_unstacked):
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
    print(list(recommendation_set))