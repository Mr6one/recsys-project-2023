import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class BPRLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()

        self.alpha = alpha

    def forward(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(axis=1)
        neg_scores = (users * neg_items).sum(axis=1)

        loss = -F.logsigmoid(pos_scores - neg_scores)
        reg = 0.5 * self.alpha * (torch.norm(users, dim=1).pow(2) + torch.norm(pos_items, dim=1).pow(2) + torch.norm(neg_items, dim=1).pow(2))

        loss = (loss + reg).mean()
        return loss
