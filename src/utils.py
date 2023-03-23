import json
import argparse

import pandas as pd
from scipy.sparse import csr_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    return config


def to_numeric_id(data, field):
    idx_data = data[field].astype('category')
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        new_index, data_index[entity] = to_numeric_id(data, field)
        data = data.assign(**{f'{field}': new_index}) 
    return data, data_index


def generate_interactions_matrix(data, data_description, rebase_users=False):
    n_users = data_description['n_users']
    n_items = data_description['n_items']

    user_idx = data[data_description['users']].values
    if rebase_users:
        user_idx, user_index = pd.factorize(user_idx, sort=True)
        n_users = len(user_index)
    item_idx = data[data_description['items']].values
    feedback = data[data_description['feedback']].values

    return csr_matrix((feedback, (user_idx, item_idx)), shape=(n_users, n_items))


def verify_time_split(before, after, target_field='userid', timeid='timestamp'):
    before_ts = before.groupby(target_field)[timeid].max()
    after_ts = after.groupby(target_field)[timeid].min()
    assert (
        before_ts
        .reindex(after_ts.index)
        .combine(after_ts, lambda x, y: True if x!=x else x <= y)
    ).all()

    
def get_table_barplot(dataset_metrics, model_names= ['ALS', 'eALS', 'iALS', 'NGCF'], stacked=True):
    metrics = ['HR', 'MRR', 'nDCG', 'COV']
    
    df_metrics = pd.DataFrame(dataset_metrics, columns=metrics, index=model_names)
    df_metrics.T.plot.bar(stacked=stacked, alpha=0.9)
    
    return df_metrics
