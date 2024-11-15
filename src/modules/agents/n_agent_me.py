import torch.nn as nn
import torch as th


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
    

class CareAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CareAgent, self).__init__()

        self.args = args
        self.n_agents   = args.n_agents
        self.n_entities = args.n_entities
        self.feat_dim  = args.obs_entity_feats - 1 # !!!!!!
        self.task_dim = args.task_feats
        self.emb_dim    = args.emb

        self.task_embedding = TaskEncoder(self.task_dim, self.emb_dim)

        self.encoder_0 = nn.Sequential(
            nn.Linear(self.n_entities * self.feat_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.encoder_1 = nn.Sequential(
            nn.Linear(self.n_entities * self.feat_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.encoder_2 = nn.Sequential(
            nn.Linear(self.n_entities * self.feat_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 3),
            nn.Softmax(dim=-1)
        ) 

        self.h_encoder = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
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
        task = task.view(-1, self.task_dim)
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim + 1)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)
        
        inputs = inputs[:, :, :-1]
        inputs = inputs.reshape(-1, self.n_entities * self.feat_dim)

        task = self.task_embedding(task)

        embs_0 = self.encoder_0(inputs)
        embs_1 = self.encoder_1(inputs)
        embs_2 = self.encoder_2(inputs)
        attention = self.attention(task)

        embs = embs_0 * attention[:, 0:1] + embs_1 * attention[:, 1:2] + embs_2 * attention[:, 2:3]

        h = th.concat((task, embs), dim=-1)
        h = self.h_encoder(h)

        # get the q values
        q = self.q_basic(h)

        return q.view(b, a, -1), h.view(b, a, -1)






