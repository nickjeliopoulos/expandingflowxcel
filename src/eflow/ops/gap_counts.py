"""OP-4a  Per-gap missing counts (Eq. 25, 96; Alg. 1 L13, Alg. 2 L17).

    g_i(t) = ind_i(t) - ind_{i-1}(t) - 1

i.e. the number of not-yet-inserted positions sitting in gap i, where gap i is
the slot between active tokens i-1 and i.  Equivalently (Eq. 95-96): every
inactive position j is assigned to gap gamma(j) = sum_{k<j} a_k, and g is the
histogram of gamma over inactive j.

Output is [B, L+1] (there are d(s)+1 gaps, padded to L+1).

Cost: trivial arithmetically, but the naive form is a cumsum + a scatter_add +
a zeros allocation, i.e. three launches on a tensor small enough that launch
latency is ~100% of the runtime.  At 1-step sampling this sits directly on the
critical path.  Fuse into one kernel, or hoist into the insertion head epilogue.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register


@register("gap_counts", "reference", reference=True)
def gap_counts_reference(active: torch.Tensor) -> torch.Tensor:
    """Explicit double loop. O(B*L) python -- only ever call this on tiny B, L."""
    B, L = active.shape
    out = torch.zeros(B, L + 1, dtype=torch.long, device=active.device)
    for b in range(B):
        gap = 0
        for j in range(L):
            if active[b, j]:
                gap += 1
            else:
                out[b, gap] += 1
    return out


@register("gap_counts", "torch")
def gap_counts_torch(active: torch.Tensor) -> torch.Tensor:
    B, L = active.shape
    gamma = torch.cumsum(active.long(), dim=-1) - active.long()  # sum_{k<j} a_k
    out = torch.zeros(B, L + 1, dtype=torch.long, device=active.device)
    out.scatter_add_(-1, gamma, (~active).long())
    return out


@register("gap_counts", "compile")
@torch.compile(dynamic=False)
def gap_counts_compile(active):
    return gap_counts_torch(active)
