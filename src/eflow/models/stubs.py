"""Shape-faithful stubs for backbones that are NOT contributions of this paper.

The GeoDiff dual-encoder (E.1) and the DeFoG graph transformer (E.2) are
borrowed architectures. Benchmarking them as if they were EFM contributions
would be misleading, and reimplementing them faithfully is a week of work with
no payoff for kernel design.

So: stubs that produce the right shapes with roughly the right cost profile,
clearly labelled. Swap in the real thing only if an end-to-end number is needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeoDiffEncoderStub(nn.Module):
    """global SchNet (6 interactions) + local GIN (4 convs), hidden 128,
    edge_order=3, radius 10 A. Emits a per-edge invariant scalar that
    eq_transform maps to a per-node equivariant velocity; the two are summed."""

    def __init__(self, d=128, n_global=6, n_local=4):
        super().__init__()
        self.g = nn.ModuleList(nn.Linear(d, d) for _ in range(n_global))
        self.l = nn.ModuleList(nn.Linear(d, d) for _ in range(n_local))
        self.out = nn.Linear(d, 1)

    def forward(self, node_feat, edge_index, edge_attr):
        raise NotImplementedError("stub: implement only if end-to-end is needed")


class DeFoGGraphTransformerStub(nn.Module):
    """9 layers, (dX, dE, dy) = (256, 64, 64), 8 heads, FFN (256, 128, 128),
    RRWP structural features with k=12."""

    def __init__(self, dX=256, dE=64, dy=64, layers=9, heads=8):
        super().__init__()
        self.layers = layers
        raise NotImplementedError("stub: implement only if end-to-end is needed")
