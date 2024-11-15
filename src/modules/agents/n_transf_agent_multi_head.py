import torch.nn as nn
import torch.nn.functional as F
import torch as th

from ..layer.transformer import Transformer
    

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

        # transformer block
        self.transformer = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )

        self.q_basic_1 = nn.Linear(args.emb, args.n_actions)
        self.q_basic_2 = nn.Linear(args.emb, args.n_actions)
        self.q_basic_3 = nn.Linear(args.emb, args.n_actions)
        self.q_basic_4 = nn.Linear(args.emb, args.n_actions)
        self.q_basic_5 = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):
        # prepare the inputs 
        b, a, _ = inputs.size() # batch_size, agents, features
        task = inputs[:, :, -self.task_dim:]
        inputs = inputs[:, :, :-self.task_dim]
        task = task.view(-1, 1, self.task_dim)
        task_type = task[:, :, 3:8]
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim + 1)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)
        
        mask = inputs[:, :, -1:]
        inputs = inputs[:, :, :-1]

        embs = self.feat_embedding(inputs)

        x = th.cat((hidden_state, embs), 1)
        mask = th.cat((th.ones_like(hidden_state[:, :, 0:1]), mask), 1)

        embs = self.transformer.forward(x, x, mask)

        h = embs[:, 0:1, :]

        # get the q values
        q1 = self.q_basic_1(h)
        q2 = self.q_basic_2(h)
        q3 = self.q_basic_3(h)
        q4 = self.q_basic_4(h)
        q5 = self.q_basic_5(h)

        # 根据task_type选择q
        # task_type是one-hot的
        q = th.cat((q1, q2, q3, q4, q5), dim=-1).view(-1, 1, 5, 5)
        task_type = task_type.unsqueeze(-1).repeat(1, 1, 1, self.args.n_actions)
        q = th.sum(q * task_type, dim=-2)

        return q.view(b, a, -1), h.view(b, a, -1)






