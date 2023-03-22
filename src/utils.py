import json
import argparse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from polara import get_movielens_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    return config


def timepoint_split(data, time_split_q=0.95):
    """
    Split data into training, testset, and holdout datasets based on a timepoint split
    and according to the `warm-start` evaluation strategy.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset containing columns `userid`, `movieid`, and `timestamp`.
    time_split_q : float, optional
        The quantile value used to split the dataset based on the `timestamp` column.
        Default is 0.95.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple of three pandas DataFrames: training, testset, and holdout.
        `training` is a subset of `data` used for training the recommender system.
        `testset` is a subset of `data` used for generating recommendations for the test users.
        `holdout` is a subset excluded from `testset` containing only the most recent interactions for each test user.

    Notes
    -----
    The function splits the input `data` into three subsets: `training`, `testset`, and `holdout`.
    The split is performed based on the `timestamp` column of `data`, using `time_split_q` as the quantile value.
    The `holdout` dataset contains only the immediate interactions following the fixed timepoint for each test user from the `testset`.
    The set of users in `training` is disjoint with the set of users in the `testset`, which implements the `warm-start` scenario.
    """    
    timepoint = data.timestamp.quantile(q=time_split_q, interpolation='nearest')
    test_ = data.query('timestamp >= @timepoint')
    rest_ = data.drop(test_.index)
    holdout_ = (
        test_
        .sort_values('timestamp')
        .drop_duplicates(subset=['userid'], keep='first')
    )
    # the holdout dataframe contains interactions closest to certain timepoint from the right,
    # i.e., the corresponding items are the first in each test user profile after this timepoint
    training = rest_.query('userid not in @holdout_.userid')
    train_items = training.movieid.unique()
    testset_ = rest_.query('userid in @holdout_.userid and movieid in @train_items')
    test_users = testset_.userid.unique()
    holdout = holdout_.query(
        # if user is not in `test_users` then no evluation is possible,
        # if item is not in `train_items` it's cold start -> must be excluded
        'userid in @test_users and movieid in @train_items'
    ).sort_values('userid')
    testset = testset_.query(
        # make sure testset and holdout contain the same set of users
        'userid in @holdout.userid'
    ).sort_values('userid')
    
    return training, testset, holdout


def transform_data(train, test, holdout=None, userid='userid', itemid='movieid'):
    train_new, data_index = transform_indices(train, 'userid', 'movieid')
    testset_new = reindex_data(test, data_index, fields='items')
    if holdout is None:
        return train_new, testset_new, data_index
    holdout_new = reindex_data(holdout, data_index, fields='items')
    
    return train_new, testset_new, holdout_new, data_index


def transform_indices(data, users, items):
    '''
    Reindex columns that correspond to users and items.
    New index is contiguous starting from 0.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data to be reindexed.
    users : str
        The name of the column in `data` that contains user IDs.
    items : str
        The name of the column in `data` that contains item IDs.

    Returns
    -------
    pandas.DataFrame, dict
        The reindexed data and a dictionary with the original IDs and the new numeric IDs.
        The keys of the dictionary are 'users' and 'items'.
        The values of the dictionary are pandas Index objects.

    Examples
    --------
    >>> data = pd.DataFrame({'users': ['A', 'B', 'C'], 'items': ['X', 'Y', 'Z'], 'rating': [1, 2, 3]})
    >>> data_reindexed, data_index = transform_indices(data, 'users', 'items')
    >>> data_reindexed
       users  items  rating
    0      0      0       1
    1      1      1       2
    2      2      2       3
    >>> data_index
    {'users': Index(['A', 'B', 'C'], dtype='object'), 'items': Index(['X', 'Y', 'Z'], dtype='object')}
    '''
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) # makes a copy of dataset!
    return data, data_index


def to_numeric_id(data, field):
    """
    This function takes in two arguments, data and field. It converts the data field
    into categorical values and creates a new contiguous index. It then creates an
    idx_map which is a renamed version of the field argument. Finally, it returns the
    idx and idx_map variables. 
    """
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def reindex_data(data, data_index, fields=None):
    '''
    Reindex provided data with the specified index mapping.
    By default, will take the name of the fields to reindex from `data_index`.
    It is also possible to specify which field to reindex by providing `fields`.
    '''
    if fields is None:
        fields = data_index.keys()
    if isinstance(fields, str): # handle single field provided as a string
        fields = [fields]
    for field in fields:
        entity_name = data_index[field].name
        new_index = data_index[field].get_indexer(data[entity_name])
        data = data.assign(**{f'{entity_name}': new_index}) # makes a copy of dataset!
    return data


def generate_interactions_matrix(data, data_description, rebase_users=False):
    '''
    Convert pandas dataframe with interactions into a sparse matrix.
    Allows reindexing user ids, which help ensure data consistency
    at the scoring stage (assumes user ids are sorted in scoring array).
    '''
    n_users = data_description['n_users']
    n_items = data_description['n_items']
    # get indices of observed data
    user_idx = data[data_description['users']].values
    if rebase_users:
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data[data_description['items']].values
    feedback = data[data_description['feedback']].values
    # construct rating matrix
    return csr_matrix((feedback, (user_idx, item_idx)), shape=(n_users, n_items))


def verify_time_split(before, after, target_field='userid', timeid='timestamp'):
    '''
    Check that items from `after` dataframe have later timestamps than
    any corresponding item from the `before` dataframe. Compare w.r.t target_field.
    Usage example: assert that for any user, the holdout items are the most recent ones.
    '''
    before_ts = before.groupby(target_field)[timeid].max()
    after_ts = after.groupby(target_field)[timeid].min()
    assert (
        before_ts
        .reindex(after_ts.index)
        .combine(after_ts, lambda x, y: True if x!=x else x <= y)
    ).all()

    
def get_table_barplot(dataset_metrics, model_names= ['ALS','eALS','iALS','Neuro']):
    metrics = ['HR','MRR','nDCG','COV']
    
    df_metrics = pd.DataFrame(dataset_metrics, columns=metrics, index=model_names)
    df_metrics.T.plot.bar(stacked=True, alpha=.9)
    
    return df_metrics
