import torch.nn as nn
import torch.nn.functional as F
import torch as th

from ..layer.transformer import Transformer
    

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
    

class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()

        self.args = args
        self.n_agents   = args.n_agents
        self.n_entities = args.n_entities
        self.feat_dim  = args.obs_entity_feats - 1 # !!!!!!
        self.task_dim = args.task_feats
        self.emb_dim    = args.emb

        self.task_embedding = TaskEncoder(self.task_dim, self.emb_dim)

        # embedder
        self.feat_embedding = nn.Linear(
            self.feat_dim,
            self.emb_dim
        )

        # transformer block
        self.transformer0 = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )
        self.transformer1 = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )
        self.transformer2 = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 3),
            nn.Softmax(dim=-1)
        ) 

        self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):
        # prepare the inputs 
        b, a, _ = inputs.size() # batch_size, agents, features
        task = inputs[:, :, -self.task_dim:]
        task = task.view(-1, self.task_dim)
        inputs = inputs[:, :, :-self.task_dim]
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim + 1)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)
        
        mask = inputs[:, :, -1:]
        inputs = inputs[:, :, :-1]

        embs = self.feat_embedding(inputs)
        task = self.task_embedding(task)

        x = th.cat((hidden_state, embs), 1)
        mask = th.cat((th.ones_like(hidden_state[:, :, 0:1]), mask), 1)

        embs0 = self.transformer0.forward(x, x, mask)
        embs1 = self.transformer1.forward(x, x, mask)
        embs2 = self.transformer2.forward(x, x, mask)
    
        h0 = embs0[:, 0:1, :].squeeze(1)
        h1 = embs1[:, 0:1, :].squeeze(1)
        h2 = embs2[:, 0:1, :].squeeze(1)

        attention = self.attention(task)

        h = h0 * attention[:, 0:1] + h1 * attention[:, 1:2] + h2 * attention[:, 2:3]

        # get the q values
        q = self.q_basic(h)

        return q.view(b, a, -1), h.view(b, a, -1)






