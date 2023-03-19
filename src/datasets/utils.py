import pandas as pd
from src.utils import get_movielens_data, generate_interactions_matrix, verify_time_split, transform_indices
from polara.preprocessing.dataframes import leave_one_out, reindex


def load_dataset(dataset, data_path):

    if dataset == 'yelp':
        data = pd.read_csv(f'{data_path}/yelp.rating', sep='\t', names='userid,movieid,rating,timestamp'.split(','))
    elif dataset == 'movielens':
        data = get_movielens_data(include_time=True)
    else:
        raise ValueError('Dataset may be either "yelp" or "movielens"')
    
    training_, holdout_ = leave_one_out(
        data,
        target='timestamp',
        sample_top=True,
        random_state=0
    )

    verify_time_split(training_, holdout_)

    training, data_index = transform_indices(training_, 'userid', 'movieid')
    holdout = reindex(holdout_, data_index.values(), filter_invalid=False)
    holdout = holdout.sort_values('userid')

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        feedback = 'rating',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
    )

    userid = data_description['users']
    seen_idx_mask = training[userid].isin(holdout[data_index['users'].name].values)
    testset = training[seen_idx_mask]
    
    training_matrix = generate_interactions_matrix(training, data_description)

    return training, testset, holdout, training_matrix, data_description
