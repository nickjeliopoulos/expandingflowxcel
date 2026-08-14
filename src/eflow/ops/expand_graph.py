r"""OP-3g  Graph expand operator (App. D.5, Eq. 97-98).

An inserted node carries a node coordinate *and* a full row and column of edges.
The operator scatters the old state into the enlarged layout:

    x^eps[i]    = x_s[i]         i in A          |  z ~ N(0, sigma^2 I_Vx)  i in N
    E^eps[i,j]  = E_s[i,j]       i,j in A        |  e ~ N(0, sigma^2 I_Ve)  otherwise

then symmetrises and zeroes the diagonal.

The structure is the point: **only the A x A block moves.**  The A x N, N x A and
N x N blocks are all pure noise.  A kernel should therefore write noise
procedurally over the whole [L, L] canvas and copy only the A x A sub-block --
traffic O(|A|^2) instead of O(L^2), which matters precisely in the early,
small-|A| regime where a naive implementation wastes the most.

Permutation invariance (D.5, last paragraph) gives a free simplification: since
a graph has no inherent node ordering, per-gap counts are only needed to define
the *training target*; at sampling time any placement of the new nodes yields
the same graph up to relabelling.  So the sampling-path scatter can place all
new nodes contiguously at the end -- a pure append, no interleaving, no gather.
Verify this equivalence in tests/test_graph_equivariance.py before relying on it.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register
from eflow.ops.edge_ops import symmetrize_torch


@register("expand_graph", "torch")
def expand_graph_torch(x, E, active, counts, sigma=1.0, *, append_only=True):
    """append_only exploits permutation invariance (see docstring)."""
    B, L, Vx = x.shape
    Ve = E.shape[-1]
    dev = x.device
    if not append_only:
        raise NotImplementedError("interleaved placement: training-target path only")

    n_s = active.sum(-1)
    n_new = counts.sum(-1)
    n_t = (n_s + n_new).clamp(max=L)
    new_active = torch.arange(L, device=dev)[None] < n_t[:, None]

    x_out = torch.randn_like(x) * sigma
    keep = active[..., None]
    x_out = torch.where(keep, x, x_out)

    E_out = torch.randn_like(E) * sigma
    blk = (active[:, :, None] & active[:, None, :])[..., None]
    E_out = torch.where(blk, E, E_out)
    return x_out, symmetrize_torch(E_out), new_active


# TODO(triton): procedural-noise canvas + A x A block copy; traffic O(|A|^2).
