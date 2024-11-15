from turtle import forward
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

from ..layer.transformer import Transformer
import time


class TaskEncoder(nn.Module):
    def __init__(self, task_dim, emb_dim) -> None:
        super().__init__()

        # self.encoder = nn.Linear(task_dim, emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(task_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, task):
        return self.encoder(task)


class TransformerMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(TransformerMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.feat_dim = args.state_entity_feats - 1 # !!!!!!
        self.task_dim = args.task_feats
        self.emb_dim = args.mixer_emb

        # embedder
        self.feat_embedding = nn.Linear(
            self.feat_dim,
            self.emb_dim
        )
        self.task_embedding = TaskEncoder(self.task_dim, self.emb_dim)

        # transformer block
        self.transformer = Transformer(
            args.mixer_emb,
            args.mixer_heads,
            args.mixer_depth,
            args.ff_hidden_mult,
            args.dropout
        )

        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        self.custom_space = args.env_args.get("state_entity_mode", True)
        
        # The hypernet weights are given by the embeddings of the transformer
        self.hyper_b2 = nn.Linear(self.emb_dim, 1)

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def init_hidden(self):
        return th.zeros(1, self.n_agents, self.args.emb).to(self.args.device)

    def forward(self, qvals, hidden_states, hyper_weights, states, obs):
        hyper_weights = hyper_weights[:, :3, :]   
        # reshape
        b, _, _ = qvals.size()

        #(batch_size, state_entities, n_tokens)
        if self.custom_space:
            task = states[:, -self.task_dim:]
            inputs = states[:, :-self.task_dim]
            inputs = inputs.reshape(b, self.n_entities, self.feat_dim + 1)
            task = task.reshape(b, 1, self.task_dim)
            mask = inputs[:, :, -1:]
            inputs = inputs[:, :, :-1]
        else:
            inputs = obs.reshape(b, self.n_agents * self.n_entities, self.feat_dim)

        # compute the embeddings
        embs = self.feat_embedding(inputs)
        task = self.task_embedding(task)

        # the keys and embeddings are the input embeddings plus the hidden embeddings
        x = th.cat((task, embs, hidden_states, hyper_weights), 1)
        # x = th.cat((embs, hidden_states, hyper_weights), 1)

        #####
        mask1 = mask.clone().detach()
        mask1 = th.cat((th.ones_like(task[:, :, 0:1]), mask1, mask1[:, :self.n_agents, :], th.ones_like(hyper_weights[:, :, 0:1])), 1)
        # mask1 = th.cat((mask1, mask1[:, :self.n_agents, :], th.ones_like(hyper_weights[:, :, 0:1])), 1)
        mask2 = mask1.repeat(1, 1, self.emb_dim)
        #####

        # transformer embeddings
        embs = self.transformer.forward(x, x, mask1)

        #####
        embs = embs.masked_fill(mask2 == 0, 0)
        #####

        # First weight matrix (batch_size, n_agents, emb) -> the hidden layers of the agents
        w1 = embs[:, -3-self.n_agents:-3, :]
        # First bias matrix (batch_size, 1, emb) -> the first hyper_weight
        b1 = embs[:, -3, :].view(-1, 1, self.emb_dim)
        
        # Second weight matrix (batch_size, emb, 1) -> second hyper_weight
        w2 = embs[:, -2, :].view(-1, self.emb_dim, 1)
        # Second bias (scalar) (batch_size, 1, 1) -> third hyper_weight @ hyper_b2
        b2 = F.relu(self.hyper_b2(embs[:, -1, :])).view(-1, 1, 1)
        
        w1 = self.pos_func(w1)
        w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1) # (b, 1, emb)
        y = th.matmul(hidden, w2) + b2 # (b, 1, 1)
        
        return y, embs[:, -3:, :]

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        elif self.qmix_pos_func == "abs":
            return th.abs(x)
        else:
            return x
