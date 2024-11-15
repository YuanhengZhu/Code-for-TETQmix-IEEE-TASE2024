import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=64, heads=8):

        super().__init__()

        self.emb_dim = emb_dim
        self.heads = heads

        self.tokeys = nn.Linear(emb_dim, emb_dim * heads, bias=False)
        self.toqueries = nn.Linear(emb_dim, emb_dim * heads, bias=False)
        self.tovalues = nn.Linear(emb_dim, emb_dim * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb_dim, emb_dim)

    def forward(self, q, k, mask_q=None, mask_k=None):
        
        h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q
        
        # get keys, queries, values
        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(k).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)
        #####
        if mask_q is None:
            mask_q = torch.ones(b, t_q, 1, device=q.device)
        if mask_k is None:
            mask_k = torch.ones(b, t_k, 1, device=k.device)
        mask_q = mask_q.unsqueeze(2).repeat(1, 1, h, 1) # (b, t_q, h, 1)
        mask_k = mask_k.unsqueeze(2).repeat(1, 1, h, 1) # (b, t_k, h, 1)
        #####

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)
        #####
        mask_q = mask_q.transpose(1, 2).contiguous().view(b * h, t_q, 1)
        mask_k = mask_k.transpose(1, 2).contiguous().view(b * h, t_k, 1)
        #####

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)
        #####
        mask = torch.bmm(mask_q, mask_k.transpose(1, 2))
        assert mask.size() == (b * h, t_q, t_k)
        #####

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t_q, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)

        return self.unifyheads(out)


class att_block(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()

        self.attention = MultiHeadAttention(emb_dim=emb_dim, heads=heads)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_mult * emb_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_dim, emb_dim)
        )

    def forward(self, q, k, mask_q, mask_k):
        att = self.attention(q, k, mask_q, mask_k)
        x = self.ln1(q + att)
        ff = self.ff(x)
        x = self.ln2(x + ff)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()
        self.self_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        
    def forward(self, embs, task, mask=None):
        embs = self.self_att1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        task = self.cross_att1(q=task, k=embs, mask_q=None, mask_k=mask)
        return task
    

class Transformer_v1(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()
        self.self_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att2 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        
    def forward(self, embs, task, mask=None):
        embs = self.cross_att1(q=embs, k=task, mask_q=mask, mask_k=None)
        embs = self.self_att1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        embs = self.cross_att2(q=task, k=embs, mask_q=None, mask_k=mask)
        return embs
    

class Transformer_v1_without_first_cab(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()
        self.self_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        # self.cross_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att2 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        
    def forward(self, embs, task, mask=None):
        # embs = self.cross_att1(q=embs, k=task, mask_q=mask, mask_k=None)
        embs = self.self_att1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        embs = self.cross_att2(q=task, k=embs, mask_q=None, mask_k=mask)
        return embs
    

class Transformer_v1_without_second_cab(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()
        self.self_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        # self.cross_att2 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        
    def forward(self, embs, task, mask=None):
        embs = self.cross_att1(q=embs, k=task, mask_q=mask, mask_k=None)
        embs = self.self_att1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        # embs = self.cross_att2(q=task, k=embs, mask_q=None, mask_k=mask)
        return embs
    

class Transformer_v1_without_sab(nn.Module):
    def __init__(self, emb_dim=64, heads=8, ff_hidden_mult=4):
        super().__init__()
        # self.self_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att1 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        self.cross_att2 = att_block(emb_dim=emb_dim, heads=heads, ff_hidden_mult=ff_hidden_mult)
        
    def forward(self, embs, task, mask=None):
        embs = self.cross_att1(q=embs, k=task, mask_q=mask, mask_k=None)
        # embs = self.self_att1(q=embs, k=embs, mask_q=mask, mask_k=mask)
        embs = self.cross_att2(q=task, k=embs, mask_q=None, mask_k=mask)
        return embs
