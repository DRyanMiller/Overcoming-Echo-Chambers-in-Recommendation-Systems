# -*- coding: utf-8 -*-

# custom.py
from scipy.spatial import distance_matrix
import pandas as pd


def test_custom():
    print("In custom module")
    return None


def cluster_centroids(clustered_data):
    """Calculates the centroids of each cluster of users"""
    cluster_centroids = clustered_data.groupby(['cluster']).agg('mean')
    return centroids


def cluster_distances_df(centroids):
    centroids_array = centroids.to_numpy()
    cluster_distance_matrix = distance_matrix(centroids_array,
                                              centroids_array, p=2)
    cluster_distance_df = pd.DataFrame(cluster_distance_matrix)
    return cluster_distance_df


def nearest_clusters(cluster_distance_df, cluster, num_nearest_clusters=2):
    sorted_distances = cluster_distance_df[cluster].sort_values(ascending=True)
    return sorted_distances[1:num_nearest_clusters+1].index.values.astype(int)


def centroid_ratings(cluster_centroids, item_factors_unstacked_transposed):
    centroid_ratings = np.dot(cluster_centroids, item_factors_unstacked_transposed)
    centroid_ratings_df = pd.DataFrame(centroid_ratings)
    centroid_ratings_df.columns = item_factors_unstacked.index
    centroid_ratings_T_df = centroid_ratings_df.transpose()
    return centroid_ratings_T_df


def top_rated_movies(cluster, centroid_ratings_T_df):
    sorted_ratings = centroid_ratings_T_df[cluster].sort_values(ascending=False)
    sorted_ratings_df = sorted_ratings.reset_index()
    most_rated_df = pd.merge(most_rated, movies_df, how='left', on='movieId')
    most_rated_df.drop(['total_recs', 0], axis=1, inplace=True)
    top_movies = pd.merge(sorted_ratings_df, most_rated_df, how='inner', left_on='id', right_on='movieId')
    top_movies.columns = ['id', 'rating', 'movieId', 'title', 'genres']
    top_10_movies = top_movies.sort_values(by='rating', ascending=False )[:10]
    return top_10_movies.title


def get_recommendations(user_cluster, clustered_data):
    centroids = cluster_centroids(clustered_data)
    cluster_distance_df = cluster_distances_df(centroids)
    near_clusters = nearest_clusters(user_cluster)
    recommendation_set = set()
    for index, cluster in enumerate(near_clusters):
        if index==0:
            recs = np.random.choice(top_rated_movies(cluster), size=6, replace=False)
            recommendation_set.update(set(recs))
        if index==1:
            cluster_unique_top_movies = set(top_rated_movies(cluster)).difference(recommendation_set)
            recs = np.random.choice(list(cluster_unique_top_movies), size=4, replace=False)
            recommendation_set.update(recs)
    print(list(recommendation_set))