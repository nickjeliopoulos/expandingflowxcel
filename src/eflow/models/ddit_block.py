"""Minimal DDiT block with EFM's per-position adaLN (App. C.3, E.3).

Config-faithful to E.3 (12 blocks, d=768, 12 heads, cond=128, dropout 0.1) but
deliberately *small by default* -- this exists so ops can be measured in context,
not so anything can be trained. There is no data loading and no training loop in
this repository by design.

The one non-standard thing is ``c`` being [B, L, 6D] rather than [B, 6D].
Everything downstream of that is standard DiT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eflow.ops.adaln import DualTimeEmbedder, modulate


class DDiTBlock(nn.Module):
    def __init__(self, d=768, n_heads=12, mlp_ratio=4, dropout=0.1, per_position=True):
        super().__init__()
        self.d, self.h, self.per_position = d, n_heads, per_position
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, mod, attn_mask=None):
        """mod: [B, L, 6D] (per-position) or [B, 1, 6D] (broadcast baseline)."""
        s1, c1, g1, s2, c2, g2 = mod.chunk(6, dim=-1)
        h = modulate(self.n1(x), s1, c1)
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(B, L, D))
        x = x + g2 * self.mlp(modulate(self.n2(x), s2, c2))
        return x


class TinyDDiT(nn.Module):
    """Shape-faithful stack. Set blocks=1 for op attribution, 12 for E.3 parity."""

    def __init__(self, V=30522, d=768, blocks=12, n_heads=12, cond=128,
                 per_position=True):
        super().__init__()
        self.embed = nn.Linear(V, d, bias=False)     # simplex-valued input, not ids
        self.time = DualTimeEmbedder(cond, d, n_mod=6)
        self.blocks = nn.ModuleList(
            DDiTBlock(d, n_heads, per_position=per_position) for _ in range(blocks))
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(d, V, bias=False)
        self.per_position = per_position

    def forward(self, x, t_local, t_target, attn_mask=None):
        h = self.embed(x)
        mod = self.time(t_local, t_target)                # [B, L, 6D]
        if not self.per_position:                          # ablation A2 baseline
            mod = mod.mean(1, keepdim=True)
        for blk in self.blocks:
            h = blk(h, mod, attn_mask)
        return self.head(self.norm_f(h))                   # logits, NOT softmaxed
