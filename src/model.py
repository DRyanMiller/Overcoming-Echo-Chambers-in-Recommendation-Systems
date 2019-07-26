# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# code for running ml model goes here.


def test_model():
    print('In Model')
    pass

def unstack(factor_df, rank):
    """Unstacks the factor arrays into pandas dataframes with factors as columns
    and users/items as rows"""
    multiplier = factor_df['id'].nunique()
    feature_array = np.array(list(range(1, rank+1))*multiplier)
    factor_df['value'] = feature_array
    factor_df_unstacked = factor_df.pivot(index='id', columns='value',
                                          values='features')
    return factor_df_unstacked