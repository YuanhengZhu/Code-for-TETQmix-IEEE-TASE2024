import torch as th
import torch.nn as nn

from ..layer.my_transformer import Transformer_v1_without_sab as MyTransformer


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

        # embedder
        self.feat_embedding = nn.Linear(
            self.feat_dim,
            self.emb_dim
        )
        self.task_embedding = TaskEncoder(self.task_dim, self.emb_dim)

        self.my_transformer = MyTransformer(
            args.emb,
            args.heads,
            args.ff_hidden_mult,
        )

        self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):
        # prepare the inputs 
        b, a, _ = inputs.size() # batch_size, agents, features
        task = inputs[:, :, -self.task_dim:]
        inputs = inputs[:, :, :-self.task_dim]
        task = task.view(-1, 1, self.task_dim)
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim + 1)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)
        
        mask = inputs[:, :, -1:]
        inputs = inputs[:, :, :-1]

        embs = self.feat_embedding(inputs)
        task = self.task_embedding(task)

        embs = self.my_transformer.forward(embs, task, mask)

        h = embs[:, 0:1, :]

        # get the q values
        q = self.q_basic(h)

        return q.view(b, a, -1), h.view(b, a, -1)






