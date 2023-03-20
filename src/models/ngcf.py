import os
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
import pytorch_lightning as pl

from src.losses import BPRLoss
from src.datasets import NGCFDataset
from src.evaluation.evaluation import downvote_seen_items, topn_recommendations, model_evaluate


class NGCF(pl.LightningModule):
    def __init__(
            self, 
            n_users, 
            n_items, 
            hidden_dim, 
            layers_dim, 
            device='cpu', 
            batch_size = 128, 
            learning_rate=3e-4,
            n_epochs=10,
            num_devices=1, 
            alpha=1e-5,
            version=None
        ):

        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items

        self.users_embed = self._create_parameter_matrix(n_users, hidden_dim)
        self.items_embed = self._create_parameter_matrix(n_items, hidden_dim)

        self.main = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(2)]) for in_dim, out_dim in zip(layers_dim[:-1], layers_dim[1:])
        ])

        self.adj_matrix = None
        self.criterion = BPRLoss(alpha=alpha)

        self.batch_size = batch_size
        self.num_devices = num_devices
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.version = version

        self.user_embeddings = nn.Parameter(torch.empty(n_users, len(layers_dim) * hidden_dim))
        self.item_embeddings = nn.Parameter(torch.empty(n_items, len(layers_dim) * hidden_dim))

        self.to(device)

    def _create_parameter_matrix(self, in_dim, out_dim):
        std = np.sqrt(1 / out_dim)
        return nn.Parameter(nn.init.uniform_(torch.empty(in_dim, out_dim), a=-std, b=std))

    def forward(self, A):
        
        embeddings = torch.cat([self.users_embed, self.items_embed])

        all_embeddings = [embeddings]
        for layers in self.main:
            lienar1, linear2 = layers

            cache = torch.sparse.mm(A, embeddings)
            embeddings = F.leaky_relu(lienar1(cache + embeddings) + linear2(cache * embeddings))

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        embeddings = torch.concat(all_embeddings, dim=1)

        user_embeddings = embeddings[:self.n_users]
        item_embeddings = embeddings[self.n_users:]

        self.user_embeddings = nn.Parameter(user_embeddings)
        self.item_embeddings = nn.Parameter(item_embeddings)

        return user_embeddings, item_embeddings
    
    def _prepare_model_input(self, R):
        R.data = np.ones_like(R.data)
        n_users, n_items = R.shape
        R = R.tolil()

        A = sp.lil_matrix((n_users + n_items, n_users + n_items))
        A[n_users:, :n_users] = R.T
        A[:n_users, n_users:] = R

        neighbours = np.array(A.sum(axis=1)).flatten()
        neighbours_inv = np.sqrt(np.divide(1, neighbours, out=np.zeros_like(neighbours), where=neighbours!=0))
        d_inv_sqrt = sp.diags(neighbours_inv)
        A = (d_inv_sqrt @ A @ d_inv_sqrt).tocoo()

        indices = torch.from_numpy(np.stack(A.nonzero())).to(torch.long)
        data = torch.from_numpy(A.data).to(torch.float32)

        A = torch.sparse_coo_tensor(indices, data)
        return A
    
    def training_step(self, batch, batch_idx):
        users, pos_items, neg_items = batch
        user_embeddings, item_embeddings = self(self.adj_matrix)

        user_embeddings_ = user_embeddings[users]
        pos_items_embeddings = item_embeddings[pos_items]
        neg_items_embeddings = item_embeddings[neg_items]
        loss = self.criterion(user_embeddings_, pos_items_embeddings, neg_items_embeddings)

        self.log('loss/training', loss, prog_bar=True, on_epoch=True, on_step=True)

        if self.holdout_data is not None:
            self._validation_step()

        return loss
    
    def _validation_step(self):
        topn = 10
        holdout, testset, data_description = self.holdout_data

        scores_movielens = self.score_users(holdout.userid.values)
        downvote_seen_items(scores_movielens, testset, data_description)
        recs_movielens = topn_recommendations(scores_movielens, topn=topn)
        hr, mrr, ndcg, _ = model_evaluate(recs_movielens, holdout, data_description, topn=topn) 

        self.log('validation/hr', hr, on_epoch=True, on_step=False)
        self.log('validation/mrr', mrr, on_epoch=True, on_step=False)
        self.log('validation/ndcg', ndcg, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def fit(self, interactions_matrix, holdout_data=None):
        device = 'gpu' if self.device.type == 'cuda' else 'cpu'
        self.holdout_data = holdout_data

        self.adj_matrix = self._prepare_model_input(interactions_matrix).to(self.device)
        dataset = NGCFDataset(interactions_matrix)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=NGCFDataset.collate_fn)

        logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=self.version, name='lightning_logs')
        trainer = pl.Trainer(devices=self.num_devices, accelerator=device, logger=logger, max_epochs=self.n_epochs)
        trainer.fit(self, dataloader)
        return self
    
    @torch.no_grad()
    def _recommend_all(self, user_ids):
        user_ids = torch.from_numpy(user_ids).to(torch.long).to(self.device)
        scores = self.user_embeddings[user_ids] @ self.item_embeddings.T
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
