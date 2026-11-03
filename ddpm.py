"""Navigation-conditioned DDPM over furniture layouts (Section 3.3, Eqs. 3-8).

x0 = per-object pose [x, y, cos(theta), sin(theta)] in the room's normalised frame.
The denoiser eps_theta(x_t, t, G*) is a transformer over objects conditioned on the GAT embedding G*,
object category and size, room type, floor-plan mask and a navigation condition (ANS, HC).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scene import CATEGORIES, ROOM_TYPES

POSE_DIM = 4


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class FloorEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(), nn.Conv2d(128, 128, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, dim))

    def forward(self, x):
        return self.net(x)


class Denoiser(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, gat_dim=128, use_gat=True):
        super().__init__()
        self.dim, self.use_gat = dim, use_gat
        self.x_in = nn.Linear(POSE_DIM, dim)
        self.cat = nn.Embedding(len(CATEGORIES), dim)
        self.size = nn.Linear(4, dim)
        self.g_in = nn.Linear(gat_dim, dim) if use_gat else None
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.room = nn.Embedding(len(ROOM_TYPES), dim)
        self.floor = FloorEncoder(dim)
        self.nav = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nav_null = nn.Parameter(torch.zeros(dim))
        enc = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, POSE_DIM))

    def forward(self, x_t, t, b, g=None, nav_drop=None):
        mask = b["mask"]
        tok = self.x_in(x_t) + self.cat(b["cats"]) + self.size(b["feats"])
        if self.use_gat and g is not None:
            tok = tok + self.g_in(g)
        nav = self.nav(b["nav"])
        if nav_drop is not None:
            nav = torch.where(nav_drop[:, None], self.nav_null[None].expand_as(nav), nav)
        cond = self.t_mlp(timestep_embedding(t, self.dim)) + self.room(b["room"]) + self.floor(b["floor"]) + nav
        tok = tok + cond[:, None]
        h = self.tf(tok, src_key_padding_mask=~mask)
        return self.out(h) * mask[..., None].float()


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        self.T = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("ac", ac)
        self.register_buffer("ac_prev", torch.cat([torch.ones(1), ac[:-1]]))

    def q_sample(self, x0, t, noise):                       # Eq. (4)
        a = self.ac[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, b, g, lambda_spatial=0.1, nav_dropout=0.1):
        x0, mask = b["pose"], b["mask"]
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0) * mask[..., None].float()
        xt = self.q_sample(x0, t, noise)
        drop = torch.rand(B, device=x0.device) < nav_dropout
        eps = model(xt, t, b, g, nav_drop=drop)
        m = mask[..., None].float()
        l_diff = ((eps - noise) ** 2 * m).sum() / (m.sum() * x0.shape[-1])          # Eq. (7)
        # Eq. (8): spatial loss on pairwise distances of the predicted clean layout
        a = self.ac[t].view(-1, 1, 1)
        x0_hat = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        p_hat, p = x0_hat[..., :2], x0[..., :2]
        d_hat = torch.cdist(p_hat, p_hat)
        d = torch.cdist(p, p)
        pm = (mask[:, :, None] & mask[:, None, :]).float()
        w = a.view(-1, 1, 1)                               # down-weight very noisy steps
        l_sp = (((d_hat - d) ** 2) * pm * w).sum() / pm.sum().clamp(min=1)
        return l_diff + lambda_spatial * l_sp, {"l_diff": l_diff.item(), "l_spatial": l_sp.item()}

    @torch.no_grad()
    def sample(self, model, b, g, guidance=2.0, return_steps=(), generator=None):
        """Ancestral sampling (Eq. 6) with classifier-free guidance on the navigation condition."""
        mask = b["mask"]
        m = mask[..., None].float()
        B, N = mask.shape
        x = torch.randn(B, N, POSE_DIM, device=mask.device, generator=generator) * m
        keep = {}
        no = torch.zeros(B, dtype=torch.bool, device=mask.device)
        yes = torch.ones(B, dtype=torch.bool, device=mask.device)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=mask.device, dtype=torch.long)
            e_c = model(x, t, b, g, nav_drop=no)
            e_u = model(x, t, b, g, nav_drop=yes)
            eps = e_u + guidance * (e_c - e_u)
            a, ab, beta = self.alphas[i], self.ac[i], self.betas[i]
            mean = (x - beta / (1 - ab).sqrt() * eps) / a.sqrt()
            if i > 0:
                var = beta * (1 - self.ac_prev[i]) / (1 - ab)
                z = torch.randn(x.shape, device=x.device, generator=generator)
                x = (mean + var.sqrt() * z) * m
            else:
                x = mean * m
            if i in return_steps:
                keep[i] = x.clone()
        x[..., :2] = x[..., :2].clamp(-1.05, 1.05)
        x[..., 2:] = F.normalize(x[..., 2:], dim=-1)
        return (x, keep) if return_steps else x
