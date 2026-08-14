"""Insertion head (App. C, D.5, E.3): reads pre-output hidden states, predicts
the per-gap two-time insertion expectation Ihat_{s,t}[i].

E.3: "reads the same 768-dimensional pre-output hidden states and is conditioned
by the same 128-dimensional time vector, adding 0.79M parameters".

Output is a *positive mean count*, so the head exponentiates (or softplus's) its
logit. Output shape [B, L+1]: there are d(s)+1 gaps.

Two training regimes appear in the paper and both must be supported, because
they have different costs:
  * E.3 (LM1B): head learns the two-time binomial interval law -> both terms of
    Eq. 26 (diagonal + off-diagonal).
  * E.2 (QM9):  diagonal term only; the two-time count is reconstructed at
    sampling time by scaling with rho_{s,t}. Cheaper; ``two_time=False``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InsertionHead(nn.Module):
    def __init__(self, d=768, cond=128, hidden=256, two_time=True):
        super().__init__()
        self.two_time = two_time
        self.cond = nn.Linear(cond, d)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.boundary = nn.Parameter(torch.zeros(1))   # the extra trailing gap

    def forward(self, h, c, rho=None):
        """h [B,L,D] pre-output hidden states; c [B,cond] time conditioning.
        Returns Ihat [B, L+1], strictly positive."""
        z = self.net(h + self.cond(c)[:, None]).squeeze(-1)          # [B, L]
        z = torch.cat([z, self.boundary.expand(z.shape[0], 1)], dim=-1)
        I = torch.nn.functional.softplus(z)
        if not self.two_time:
            # E.2: scale the diagonal head by the schedule fraction rho_{s,t}
            assert rho is not None, "diagonal-only head needs rho_{s,t}"
            I = I * rho[:, None]
        return I
