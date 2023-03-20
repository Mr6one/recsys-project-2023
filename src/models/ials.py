import numpy as np
import torch
from tqdm.notebook import tqdm


class iALS:
    def __init__(self, factors, iterations=100, alpha=1, y=1, regularization=0.1, device='cpu', callback=None):
        self.factors = factors
        self.reg = regularization
        self.iterations = iterations
        self.device = device
        self.callback = callback

        self.alpha = alpha
        self.y = y


    def fit(self, R):
        n_users, n_items = R.shape
        num_factors = self.factors

        user_factors = torch.randn(n_users, num_factors, device=self.device) / np.sqrt(num_factors)
        item_factors = torch.randn(n_items, num_factors, device=self.device) / np.sqrt(num_factors)
        I = torch.eye(self.factors, device=self.device)
        R = R.to(self.device)

        logs = []

        for _ in tqdm(range(self.iterations)):

            VV = item_factors.T @ item_factors

            for user_id in range(R.shape[0]):

                grad_w_u = 0
                grad_w_2_u = self.alpha * VV + self.reg * I

                for i, item in enumerate(R[user_id]):
                    if int(item) != 0:
                        grad_w_u += self.alpha * self.y * item_factors[i]
                        grad_w_2_u += self.alpha * item_factors[i].T @ item_factors[i]

                user_factors[user_id, :] = torch.linalg.inv(grad_w_2_u) @ grad_w_u

            UU = np.tensordot(user_factors.T, user_factors, axes=1)

            for item_id in range(R.shape[1]):
                grad_w_i = 0
                grad_w_2_i = self.alpha * UU + self.reg * I

                for j, user in enumerate(R[item_id]):
                    if int(user) != 0:
                        grad_w_i += self.alpha * self.y * user_factors[j]
                        grad_w_2_i += self.alpha * user_factors[j].T @ user_factors[j]

                item_factors[item_id, :] = torch.linalg.inv(grad_w_2_i) @ grad_w_i

            if self.callback is not None:
                log = self.callback(user_factors, item_factors)
                logs.append(log)
                
        self.item_factors = item_factors
        self.user_factors = user_factors
        return np.array(logs)


    def predict(self, id_user, k=None):
        if k is None:
            k = len(self.item_factors)

        p = self.user_factors[id_user]

        scores = self.item_factors @ p

        top_k = torch.argsort(scores, descending=True)[:k]

        return top_k
