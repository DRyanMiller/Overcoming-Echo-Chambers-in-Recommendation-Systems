# -*- coding: utf-8 -*-
# pandas operators
# wrappers for various pandas methods
import pandas as pd
import numpy as np
import operator
from collections import Counter

# file contains useful functions to access various pandas methods.
# in most cases these are wrappers that simplify the syntax

def test_pandas():
    print("In pandas ops")
    return None

## some python wrappers to manipulate lists
def unpack(data):
    """ takes data array, assumes list of tuples or lists and unpacks.
    Should be size independent
    args:
        data : list or array of data
    returns:
        out : array of unpacked data
    """
    out = []
    for items in data:
        [out.append(item) for item in items]
    return np.array(out)

def gt_(inp, val):
    return operator.gt(inp, val)


def lt_(inp, val):
    return operator.lt(inp, val)


def ge_(inp, val):
    return operator.ge(inp, val)


def le_(inp, val):
    return operator.le(inp, val)


def eq_(inp, val):
    return operator.eq(inp, val)


def ne_(inp, val):
    return operator.ne(inp, val)


# shorthand to test a boolean condition
def get_relation(inp, rel, val):
    return rel(inp, val)


## Dataframe helper routines
def concat_df(df1, df2, axis=0):
    """ helper function to join two dataframes
        default join axis = 0.
    """
    return pd.concat([df1, df2], axis=axis)


def add_df_column(df, name, value):
    """Add column with colname and value to dataframe
    value can be dataframe, nparray or constant.
    """
    df.loc[:, name] = value


def insert_df_column(df, loc, cname, value):
    """Insert col at loc with cname and value"""
    df.insert(loc, cname, value)


def drop_df_columns(df, col_list, inpl=True):
    """drop columns specified in col_list. col_list must be passed
       as list of column names. Default is inplace, so this
       this will change the passed dataframe directly. """
    df.drop(columns=col_list, inplace=inpl)


def drop_df_duplicates(df, subset=None, keep=False, inpl=True):
    """Drop duplicate rows, default is drop all, test on all row values,
       and inpl=True. If passed a column name list it will test
       duplication only on those values."""
    df.drop_duplicates(subset=subset, keep=keep, inplace=inpl)


def df_head(df, n=5):
    """ Get head of dataframe with n rows"""
    df.head(n)


def df_tail(df, n=5):
    """ Get head of dataframe with n rows"""
    df.tail(n)


## set boolean conditions
def df_isna(df):
    df.isna()


def get_bool_cond(df, rel, value):
    bool_df = get_relation(df, rel, value)
    return bool_df


def df_fillna(df, value):
    """Fill all NaN entries with value"""
    return df.fillna(value)


def df_col_dropna(df, N):
    """drop columns with more than threshold% NaN
       0 < N < 1
    """
    df.dropna(thresh=int(df.shape[0] * N), axis=1)


# string, date manipulation


def df_col_numeric(df, colname):
    pd.to_numeric(colname)


def where_cond(cond, _then, _else):
    return np.where(cond, _then, _else)


def contains_str(ser, string):
    """ return boolean index based on
        series containing string, pass
        as dataframe column or series"""
    return ser.str.contains(string)


def rename_cols_df(df, colnames, inpl=True):
    df.rename(columns=colnames, inplace=inpl)

def reorder_cols_df(df, col_order, inpl=True):
    """reorder columns in dataframe """
    cols = list(df.columns)
    check =  all(item in cols for item in col_order)
    if check:
        newdf = df[col_order]
        if inpl==True:
            df = newdf
            return 1
        else:
            return newdf 
    else:
        print('Error: Columns must be in dataframe')
        return 0


    


# Data Analysis function definitions


def data_summary(df_item, n):
    """Summarize data set:

    args:
        df_item: a single element of a dataframe
        n: integer to find n common items.
    Finds:
        most common
        least common
        avg number of samples
        computes pmf and cdf for data
        returns
    """
    item_counter = Counter(df_item)
    items, counts = zip(*item_counter.items())

    # Find most and least common labels in data
    most = item_counter.most_common(n)
    least = item_counter.most_common()[-n:]

    uniqw, inverse = np.unique(df_item, return_inverse=True)
    bins = np.bincount(inverse)
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / min_count
    avg = sum(counts) / len(items)

    return items, counts, most, least, ratio, avg, bins, uniqw


def new_data_summary(df_item):
    df_count = df_item.value_counts()  # data_frame of counts of values
    uniqw, invers = np.unique(df_item, return_inverse=True)

    return df_count, uniqw, invers 


