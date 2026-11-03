"""Graph Attention Network with edge features (Section 3.2, Eq. 1, Algorithm 1 / Algorithm 3 step 2).

h_i^(l+1) = sigma( W1 h_i^(l) + sum_{j in N(i)} alpha_ij W2 h_j^(l) )
alpha_ij  = softmax_j( LeakyReLU( a^T [W2 h_i || W2 h_j || We e_ij] ) )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import EDGE_DIM
from ..scene import CATEGORIES


class EdgeGATLayer(nn.Module):
    def __init__(self, dim, heads, edge_dim=EDGE_DIM):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.h, self.dh = heads, dim // heads
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.We = nn.Linear(edge_dim, dim, bias=False)
        self.a = nn.Parameter(torch.randn(heads, 3 * self.dh) * 0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, adj, ef):
        B, N, D = h.shape
        H, dh = self.h, self.dh
        wh = self.W2(h).view(B, N, H, dh)
        we = self.We(ef).view(B, N, N, H, dh)
        s_i = (wh * self.a[:, :dh]).sum(-1)                       # B,N,H
        s_j = (wh * self.a[:, dh:2 * dh]).sum(-1)                 # B,N,H
        s_e = (we * self.a[:, 2 * dh:]).sum(-1)                   # B,N,N,H
        score = F.leaky_relu(s_i[:, :, None, :] + s_j[:, None, :, :] + s_e, 0.2)
        m = adj.bool()[..., None]                                 # B,N,N,1
        score = score.masked_fill(~m, float("-inf"))
        alpha = torch.softmax(score, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0) * m.float()      # nodes without neighbours
        msg = torch.einsum("bijh,bjhd->bihd", alpha, wh).reshape(B, N, D)
        out = F.elu(self.W1(h) + msg)
        return self.norm(h + out), alpha


class GAT(nn.Module):
    def __init__(self, hidden=128, heads=8, layers=2, cat_dim=32, feat_dim=4):
        super().__init__()
        self.cat = nn.Embedding(len(CATEGORIES), cat_dim)
        self.inp = nn.Linear(cat_dim + feat_dim, hidden)
        self.layers = nn.ModuleList([EdgeGATLayer(hidden, heads) for _ in range(layers)])
        self.hidden = hidden
        # pre-training heads (relation type and pairwise distance)
        self.rel_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.dist_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, cats, feats, adj, ef, mask):
        h = self.inp(torch.cat([self.cat(cats), feats], -1))
        adj = adj & mask[:, None, :] & mask[:, :, None]
        for layer in self.layers:
            h, _ = layer(h, adj, ef)
        return h * mask[..., None].float()          # G* = {h_i^(L)}

    def pair_logits(self, h):
        B, N, D = h.shape
        hp = torch.cat([h[:, :, None].expand(B, N, N, D), h[:, None, :].expand(B, N, N, D)], -1)
        return self.rel_head(hp), self.dist_head(hp).squeeze(-1)
