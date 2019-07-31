# -*- coding: utf-8 -*-
# make_data.py

# General Imports
import pandas as pd


def test_make_data():
    print("In make_data")
    pass


def get_agg_counts(df, groupby_var, column=None):
    """Returns dataframe with aggregate counts grouped by groupby_var.
    Parameters
    ----------
    df: dataframe
        An dataframe of user ratings to be grouped and counted.
    groupby_var: string
        The name of the column to group by.
    column: string
        The name of the column to return counts for."""
    if column is None:
        counts = df.groupby([groupby_var]).agg('count')
        counts.drop(['rating', 'timestamp'], axis=1, inplace=True)
        counts.columns = ['counts']
    else:
        counts = df.groupby([groupby_var])[column].agg('count')
        counts.columns = ['counts']
    return counts


def filter_dataframe(df, groupby_var, inequality_type, column=None, value=0):
    """Returns dataframe with aggregate counts grouped by groupby_var.
    Parameters
    ----------
    df: dataframe
        An dataframe of user ratings to be grouped and counted.
    groupby_var: string
        The name of the column to group by.
    column: string
        The name of the column to return counts for.
    inequality: string
        The inequality used to filter data (i.e., >, <, >=, <=).
    value: int
        The value use with the inequality to filter.
    filter_var: string
        The variable name used to filter (e.g., userId_y or movieId_y"""
    filter_var = column + '_y'
    if inequality_type == '>':
        filter_ = get_agg_counts(df, groupby_var, column=column) > value
    elif inequality_type == '>=':
        filter_ = get_agg_counts(df, groupby_var, column=column) >= value
    elif inequality_type == '<':
        filter_ = get_agg_counts(df, groupby_var, column=column) < value
    elif inequality_type == '<=':
        filter_ = get_agg_counts(df, groupby_var, column=column) <= value
    filter_ = filter_.reset_index()
    ratings_with_filter = df.merge(filter_, on=groupby_var, how='left')
    try:
        ratings_with_filter.drop(['timestamp'], axis=1, inplace=True)
    except:
        pass
    ratings_filtered = ratings_with_filter[ratings_with_filter[filter_var] == True]
    ratings_filtered.drop([filter_var], axis=1, inplace=True)
    ratings_filtered.columns = ['userId', 'movieId', 'rating']
    return ratings_filtered


def get_factors(client, bucket, key, num_of_files):
    """Returns a single dataframe of factors. Used to combine the multiple
    csv files generated during the AWS implementation of the ALS model.
    Parameters
    ----------
    bucket: string
        Name of the S3 bucket containing the factors.
    key: string
        Path to the factor files with the last digit(s) of the part number
        replaced with {}.
    num_of_files: int
        The number of files to import."""
    factors = []
    for i in list(range(0, num_of_files)):
        obj = client.get_object(Bucket=bucket,
                                Key=key.format(i))
        factor_df = pd.read_csv(obj['Body'], header=None)
        print('File {} has {} rows.'.format(i, len(factor_df)))
        factors.append(factor_df)
    factors_combined = pd.concat(factors, axis=0, ignore_index=True)
    factors_combined.columns = ['id', 'features']
    return factors_combined
