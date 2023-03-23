import numpy as np
from tqdm.notebook import tqdm

import torch
from torch_scatter import scatter_sum

from src.models.base import BaseModel


class iALS(BaseModel):
    def __init__(self, factors=100, iterations=100, alpha=1, y=1, regularization=0.1, device='cpu', callback=None, save_path=None):
        
        self.factors = factors
        self.reg = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback
        self.save_path = save_path

        self.alpha = alpha
        self.y = y
        self._logs = []

        self.item_factors = None
        self.user_factors = None

    def fit(self, R):
        n_users, n_items = R.shape
        num_factors = self.factors

        user_factors = torch.randn(n_users, num_factors, device=self.device) / np.sqrt(num_factors)
        item_factors = torch.randn(n_items, num_factors, device=self.device) / np.sqrt(num_factors)
        I = torch.eye(self.factors, device=self.device)

        R = R.tocoo()
        rows, cols = torch.from_numpy(np.stack(R.nonzero())).to(torch.long)

        users2items = torch.from_numpy(R.tocsr().indices).to(torch.long)
        items2users = torch.from_numpy(R.T.tocsr().indices).to(torch.long)
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        users2items = users2items.to(self.device)
        items2users = items2users.to(self.device)

        self._logs = []
        for _ in tqdm(range(self.iterations)):

            grad_w_2_u = []
            for user_id in range(n_users):
                items = users2items[user_id: user_id + 1]
                grad_w_2_u.append(item_factors[items].T @ item_factors[items])

            grad_w_2_u = torch.stack(grad_w_2_u)
            grad_w_2_u = self.alpha * item_factors.T @ item_factors + self.reg * I + self.alpha * grad_w_2_u

            grad_w_u = self.alpha * self.y * scatter_sum(item_factors[cols], rows, dim=0)
            user_factors = torch.matmul(torch.linalg.inv(grad_w_2_u), grad_w_u.unsqueeze(2)).squeeze(2)

            grad_w_2_i = []
            for item_id in range(n_items):
                users = items2users[item_id: item_id + 1]
                grad_w_2_i.append(user_factors[users].T @ user_factors[users])

            grad_w_2_i = torch.stack(grad_w_2_i)
            grad_w_2_i = self.alpha * user_factors.T @ user_factors + self.reg * I + self.alpha * grad_w_2_i

            grad_w_i = self.alpha * self.y * scatter_sum(user_factors[rows], cols, dim=0)
            item_factors = torch.matmul(torch.linalg.inv(grad_w_2_i), grad_w_i.unsqueeze(2)).squeeze(2)

            if self.callback is not None:
                log = self.callback(user_factors, item_factors)
                self._logs.append(log)
                
        self.item_factors = item_factors
        self.user_factors = user_factors
        torch.cuda.empty_cache()

        if self.save_path is not None:
            self._save_model(self.save_path)

        return self

    def _to_device(self, device):
        self.device = device

        if self.item_factors is not None and self.user_factors is not None:
            self.item_factors = self.item_factors.to(device)
            self.user_factors = self.user_factors.to(device)

    @torch.no_grad()
    def _recommend_all(self, user_ids):
        user_ids = np.array(user_ids)
        user_ids = torch.from_numpy(user_ids).to(torch.long).to(self.device)
        scores = self.user_factors[user_ids] @ self.item_factors.T
        return scores
