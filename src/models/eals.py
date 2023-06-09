import numpy as np
from tqdm.notebook import tqdm

import torch
from torch_scatter import scatter_sum

from src.models.base import BaseModel


class eALS(BaseModel):
    def __init__(self, factors=100, iterations=100, w=1, c=1, regularization=0.1, device='cpu', callback=None, save_path=None):
        self.w = w
        self.c = c
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.device = device
        self.callback = callback
        self.save_path = save_path
        self._logs = []

        self.P = None
        self.Q = None
        
    def fit(self, X):
        K = self.factors
        M, N = X.shape

        self.P = torch.randn(M, K, device=self.device) / np.sqrt(K)
        self.Q = torch.randn(N, K, device=self.device) / np.sqrt(K)
        
        X = X.tocoo()
        rows, cols = torch.from_numpy(np.stack(X.nonzero())).to(torch.long)
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        c = self.c
        if isinstance(c, int):
            c = torch.ones(N, device=self.device) * c
        else:
            c = torch.tensor(c, device=self.device)
        
        R_hat = (self.P[rows] * self.Q[cols]).sum(axis=1)
        
        self._logs = []
        for _ in tqdm(range(self.iterations)):
            S_q = (c.unsqueeze(1) * self.Q).T @ self.Q

            for f in range(K):
                r_hat = R_hat - self.P[rows, f] * self.Q[cols, f]

                nominator = scatter_sum((self.w - (self.w - c[cols]) * r_hat) * self.Q[cols, f], rows)
                nominator -= self.P @ S_q[:, f] - self.P[:, f] * S_q[f, f]

                denominator = scatter_sum((self.w - c[cols]) * self.Q[cols, f] ** 2, rows) + S_q[f, f]
                self.P[:, f] = nominator / (denominator + self.regularization)

                R_hat = r_hat + self.P[rows, f] * self.Q[cols, f]

            S_p = self.P.T @ self.P
            for f in range(K):
                r_hat = R_hat - self.P[rows, f] * self.Q[cols, f] 

                nominator = scatter_sum((self.w - (self.w - c[cols]) * r_hat) * self.P[rows, f], cols)
                nominator -= c * (self.Q @ S_p[:, f] - self.Q[:, f] * S_p[f, f])

                denominator = scatter_sum((self.w - c[cols]) * self.P[rows, f] ** 2, cols) + c * S_p[f, f] 
                self.Q[:, f] = nominator / (denominator + self.regularization)

                R_hat = r_hat + self.P[rows, f] * self.Q[cols, f]
                
            if self.callback is not None:
                log = self.callback(self.P, self.Q)
                self._logs.append(log)

        torch.cuda.empty_cache()

        if self.save_path is not None:
            self._save_model(self.save_path)

        return self

    def _to_device(self, device):
        self.device = device

        if self.P is not None and self.Q is not None:
            self.P = self.P.to(device)
            self.Q = self.Q.to(device)
    
    @torch.no_grad()
    def _recommend_all(self, user_ids):
        user_ids = np.array(user_ids)
        user_ids = torch.from_numpy(user_ids).to(torch.long).to(self.device)
        scores = self.P[user_ids] @ self.Q.T
        return scores
