r"""OP-10  Continuous child expansion + COM projection (Sec. 3(iii), App. E.1).

    x0_i = x0_{pa(i)} + sigma_h * eps          child expansion, sigma_h = 0.1
    x   <- x - mean(x)  per molecule           "center-of-mass free at every step"

The conformer path's expand operator is *deterministic* -- the molecular graph
fixes the atom count, hydrogens attach to a known heavy parent, and insertion
times are drawn once before integration.  No learned head, no count inference,
no sampling-time cost.  So the novel operators here are only these two, plus
per-atom local-time Fourier embeddings gathered onto edges.

The COM projection is the one with real systems content: it is a *segmented*
mean over a ragged batch of molecules, applied every step, on [sum_A, 3].
Three implementations worth racing (bench_ablations.py):
    dense-padded mean with a mask        -- wastes on Drugs' 181-atom tail
    scatter_mean / index_add over CSR    -- atomics, contention on big molecules
    segmented reduction with per-molecule blocks -- one block per molecule

The GeoDiff dual-encoder itself (global SchNet + local GIN, App. E.1) is NOT
novel to this paper and is deliberately a shape-faithful stub in models/stubs.py.
Do not benchmark it as if it were an EFM contribution.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register


@register("child_expand", "reference", reference=True)
def child_expand_reference(x0_heavy, parent, sigma_h=0.1, *, eps=None):
    A = parent.shape[-1]
    out = torch.zeros(*parent.shape, 3, dtype=torch.float64, device=parent.device)
    for b in range(parent.shape[0]):
        for i in range(A):
            p = int(parent[b, i])
            e = eps[b, i] if eps is not None else torch.randn(3, device=parent.device)
            out[b, i] = x0_heavy[b, p].double() + sigma_h * e.double()
    return out


@register("child_expand", "torch")
def child_expand_torch(x0_heavy, parent, sigma_h=0.1, *, eps=None):
    base = torch.gather(x0_heavy, 1, parent[..., None].expand(-1, -1, 3))
    if eps is None:
        eps = torch.randn_like(base)
    return base + sigma_h * eps


@register("com_free", "torch")
def com_free_dense(x, mask):
    """Dense padded segmented mean. mask [B, A]."""
    m = mask[..., None].to(x.dtype)
    mean = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    return (x - mean) * m


@register("com_free", "reference", reference=True)
def com_free_reference(x, mask):
    out = x.double().clone()
    for b in range(x.shape[0]):
        idx = mask[b].nonzero().squeeze(-1)
        out[b, idx] = out[b, idx] - out[b, idx].mean(0, keepdim=True)
        out[b, ~mask[b]] = 0
    return out


@register("com_free", "segment")
def com_free_segment(x_flat, batch_idx, n_mol):
    """Ragged CSR form: x_flat [N, 3], batch_idx [N]. index_add + gather."""
    cnt = torch.zeros(n_mol, device=x_flat.device, dtype=x_flat.dtype)
    cnt.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=x_flat.dtype))
    s = torch.zeros(n_mol, 3, device=x_flat.device, dtype=x_flat.dtype)
    s.index_add_(0, batch_idx, x_flat)
    return x_flat - (s / cnt.clamp_min(1)[:, None])[batch_idx]
