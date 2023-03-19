import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class NGCFDataset(Dataset):
    def __init__(self, interaction_matrix, max_items=100):
        n_users, n_items = interaction_matrix.shape

        self.n_users = n_users
        self.n_items = n_items
        self.max_items = max_items

        self.users_interactions = self._matrix_to_lists(interaction_matrix)
        self.all_items = set(np.arange(n_items))

    def _matrix_to_lists(self, interaction_matrix):
        A = interaction_matrix.tocoo()
        indices = np.stack(A.nonzero()).T
        data = pd.DataFrame(indices, columns=['users', 'items'])
        return data.groupby('users')['items'].apply(list)

    def __len__(self):
        return self.n_users
    
    def __getitem__(self, idx):
        pos_items = self.users_interactions.iloc[idx]
        neg_items = list(self.all_items.difference(pos_items))

        num_items = min(len(pos_items), len(neg_items), self.max_items)
        pos_items = np.random.choice(pos_items, size=num_items, replace=False)
        neg_items = np.random.choice(neg_items, size=num_items, replace=False)
        
        idx = np.array([idx] * num_items)
        idx, pos_items, neg_items = map(lambda x: torch.tensor(x).to(torch.long), [idx, pos_items, neg_items])

        return idx, pos_items, neg_items
    
    @staticmethod
    def collate_fn(batch):
        idx, pos_items, neg_items = zip(*batch)
        idx, pos_items, neg_items = map(lambda x: torch.concat(x), [idx, pos_items, neg_items])
        return idx, pos_items, neg_items
