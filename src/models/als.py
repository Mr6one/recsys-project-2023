import pickle
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch_scatter import scatter_sum


class ALS:
    def __init__(self, factors, iterations=100, regularization=0.1, device='cpu', callback=None, save_path=None):
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback
        self.save_path = save_path
        self._logs = []
        
    def fit(self, R, validation=None):

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
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return self
    
    @torch.no_grad()
    def _recommend_all(self, user_ids):
        user_ids = torch.from_numpy(user_ids).to(torch.long).to(self.device)
        scores = self.P[user_ids] @ self.Q.T
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
