import numpy as np
from tqdm.notebook import tqdm

import torch
from torch_scatter import scatter_sum

from src.models.base import BaseModel


class ALS(BaseModel):
    def __init__(self, factors=100, iterations=100, regularization=0.1, device='cpu', callback=None, save_path=None):
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback
        self.save_path = save_path
        self._logs = []

        self.P = None
        self.Q = None
        
    def fit(self, R):
        m, n = R.shape
        
        P = torch.randn(m, self.factors, device=self.device) / np.sqrt(self.factors)
        Q = torch.randn(n, self.factors, device=self.device) / np.sqrt(self.factors)
        I = torch.eye(self.factors, device=self.device)
        
        R = R.tocoo()
        rows, cols = torch.from_numpy(np.stack(R.nonzero())).to(torch.long)
        
        rows = rows.to(self.device)
        cols = cols.to(self.device)
        
        self._logs = []
        for _ in tqdm(range(self.iterations)):
                
            A = torch.linalg.inv(Q.T @ Q + self.regularization * I) @ Q.T
            P = scatter_sum(A[:, cols], rows).T

            A = torch.linalg.inv(P.T @ P + self.regularization * I) @ P.T
            Q = scatter_sum(A[:, rows], cols).T
            
            if self.callback is not None:
                log = self.callback(P, Q)
                self._logs.append(log)
                
        self.P = P
        self.Q = Q

        torch.cuda.empty_cache()

        if self.save_path is not None:
            self._save_model(self.save_path)

        return self
    
    def save_model(self, path):
        self._save_model(path)

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
