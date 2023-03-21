import numpy as np
from tqdm.notebook import tqdm

import torch
from torch_scatter import scatter_sum


class iALS:
    def __init__(self, factors, iterations=100, alpha=1, y=1, regularization=0.1, device='cpu', callback=None):
        self.factors = factors
        self.reg = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback

        self.alpha = alpha
        self.y = y

        self._logs = []

    def fit(self, R):
        n_users, n_items = R.shape
        num_factors = self.factors

        user_factors = torch.randn(n_users, num_factors, device=self.device) / np.sqrt(num_factors)
        item_factors = torch.randn(n_items, num_factors, device=self.device) / np.sqrt(num_factors)
        I = torch.eye(self.factors, device=self.device)

        users2items = torch.from_numpy(R.tocsr().indices).to(torch.long)
        items2users = torch.from_numpy(R.T.tocsr().indices).to(torch.long)

        R = R.tocoo()
        rows, cols = torch.from_numpy(np.stack(R.nonzero())).to(torch.long)
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        users2items = users2items.to(self.device)
        items2users = items2users.to(self.device)

        self._logs = []
        for _ in tqdm(range(self.iterations)):

            VV = item_factors.T @ item_factors
            grad_w_u = self.alpha * self.y * scatter_sum(item_factors[cols], rows, dim=0)
            for user_id in range(n_users):
                items = users2items[user_id: user_id + 1]
                grad_w_2_u = self.alpha * VV + self.reg * I + self.alpha * item_factors[items].T @ item_factors[items]
                user_factors[user_id] = torch.linalg.inv(grad_w_2_u) @ grad_w_u[user_id]

            UU = user_factors.T @ user_factors
            grad_w_i = self.alpha * self.y * scatter_sum(item_factors[rows], cols, dim=0)
            for item_id in range(n_items):
                users = items2users[item_id: item_id + 1]
                grad_w_2_i = self.alpha * UU + self.reg * I + self.alpha * user_factors[users].T @ user_factors[users]
                item_factors[item_id] = torch.linalg.inv(grad_w_2_i) @ grad_w_i[item_id]

            if self.callback is not None:
                log = self.callback(user_factors, item_factors)
                self._logs.append(log)
                
        self.item_factors = item_factors
        self.user_factors = user_factors

        return self

    @torch.no_grad()
    def _recommend_all(self, user_ids):
        user_ids = torch.from_numpy(user_ids).to(torch.long).to(self.device)
        scores = self.user_factors[user_ids] @ self.item_factors.T
        return scores
    
    @torch.no_grad()
    def recommend(self, user_ids, N=10):
        scores = self._recommend_all(user_ids)
        output = torch.topk(scores, N, dim=1).indices
        return output.cpu().numpy()

    @torch.no_grad()
    def recommend_all(self, user_ids):
        scores = self._recommend_all(user_ids)
        output = torch.argsort(scores, descending=True, dim=1)
        return output.cpu().numpy()
    
    @torch.no_grad()
    def score_users(self, user_ids):
        scores = self._recommend_all(user_ids)
        return scores.cpu().numpy()
