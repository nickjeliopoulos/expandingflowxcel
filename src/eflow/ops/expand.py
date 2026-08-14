r"""OP-3  Sequence expand operator (Eq. 27, Alg. 3).

    x_s^eps = [ (+)_{i=1..d(s)} ( x0^{l_i} (+) x_s^i ) ] (+) x0^{l_{d(s)+1}}

Given the compacted state x_s of length n_s and per-gap counts l_i, produce the
augmented state of length n_t = n_s + sum_i l_i, where the old tokens keep their
relative order and every new slot is a fresh Gaussian latent at local time 0.

Implementation
--------------
In the fixed-buffer representation this is a *permutation of the active rows
plus a procedural noise fill of the new rows*.  The destination of old token j
is

    dest(j) = j + offset[gamma(j)],   offset = exclusive_cumsum(l)

so the whole operator is: one exclusive cumsum over [B, G], one gather of
[B, n_s] indices, one row-permutation of x, and noise into the complement.

Two observations that drive the kernel design:

1. **Only the old rows move.**  New rows are pure noise, so a kernel should
   never copy them from anywhere -- it should generate them in place, exactly
   as in OP-2.  Naive implementations build a full [B,L,V] noise tensor and then
   scatter over it, doubling traffic on the largest tensor in the method.
   Traffic floor is 1 read + 1 write of the *active* rows only.

2. **On the training path this op is nearly free and should not be called.**
   Alg. 2 L12 expands *once* to d(t) with the gaps over (s,t] left as noise --
   there is no data movement at all, only a change of activity mask.  Calling a
   real expand during training is a correctness-preserving but pointless cost.
   ``expand_training`` implements the mask-only form; ``expand_sampling``
   implements the moving form.  Benchmark them separately; conflating them is
   the most likely way to misattribute cost in this repo.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(x.cumsum(-1)[..., :-1], (1, 0))


@register("expand_sampling", "reference", reference=True)
def expand_sampling_reference(x, t_local, active, counts, sigma=1.0, *, noise=None):
    """Literal Alg. 3: rebuild the sequence gap by gap with python lists.

    Returns (x_eps, active_eps, t_local_eps). Old tokens carry their local time
    with them; every newly inserted token starts at local time 0 (Eq. 27).
    """
    B, L, V = x.shape
    out = torch.zeros_like(x, dtype=torch.float64)
    new_active = torch.zeros_like(active)
    new_tlocal = torch.zeros(B, L, dtype=torch.float64, device=x.device)
    for b in range(B):
        old = [j for j in range(L) if active[b, j]]
        slots = []                                   # source row, or None = noise
        for i in range(len(old) + 1):
            slots += [None] * int(counts[b, i])      # Alg. 3 L4-L5: l_i new tokens
            if i < len(old):
                slots.append(old[i])                 # Alg. 3 L6: then the old token
        for k, j in enumerate(slots[:L]):
            new_active[b, k] = True                  # inserted == active, t_i = 0
            if j is None:
                # NOTE: caller-supplied `noise` is treated as ALREADY scaled by
                # sigma, so that reference and fast paths are comparable when a
                # test pins the noise. Only self-drawn noise gets scaled here.
                out[b, k] = (noise[b, k] if noise is not None
                             else torch.randn(V, device=x.device) * sigma)
            else:
                out[b, k] = x[b, j].double()
                new_tlocal[b, k] = t_local[b, j]
        # Trailing padding beyond d(t) sits at its Gaussian latent with t_i = 0,
        # exactly like a not-yet-inserted position (App. C.1) -- NOT at zero.
        # The vectorised path gets this for free from its `where`; spell it out
        # here so the two agree.
        for k in range(min(len(slots), L), L):
            out[b, k] = (noise[b, k] if noise is not None
                         else torch.randn(V, device=x.device) * sigma)
    return out, new_active, new_tlocal


@register("expand_sampling", "torch")
def expand_sampling_torch(x, t_local, active, counts, sigma=1.0, *, noise=None):
    """Vectorised form. Builds an explicit source-index map, then one gather.

    src[b, k] = the row of x that lands in destination slot k, or -1 for a
    freshly inserted noise token. Inverting the destination map this way (rather
    than scattering x directly) means the noise never has to be written and then
    overwritten -- which is exactly the traffic a Triton kernel should eliminate
    entirely by generating noise procedurally at the -1 slots.
    """
    B, L, V = x.shape
    dev = x.device
    a = active.long()
    gap = a.cumsum(-1) - a                       # gap index gamma(j) of each slot
    rank = (a.cumsum(-1) - 1).clamp_min(0)       # index of j among active rows
    offset = exclusive_cumsum(counts)            # [B, G] tokens inserted before gap i
    dest = (rank + offset.gather(-1, gap.clamp(max=offset.shape[-1] - 1))).clamp(max=L - 1)

    # Build the inverse map. Inactive rows must NOT participate: scattering both
    # active and inactive rows into the same slot leaves the winner up to
    # scatter's (unspecified) duplicate-index order. Route every inactive row to
    # a scratch column at index L and slice it off.
    #
    # dest is injective on the active set: for active j1 < j2 we have
    # rank1 < rank2 and gap1 <= gap2, hence offset1 <= offset2 and dest1 < dest2.
    # So the active writes never collide, provided counts respect the budget
    # (sum_i l_i <= L - n_s, guaranteed by insertion_sample). The clamp below is
    # a safety net for a caller that violates that, not part of the contract.
    src = torch.full((B, L + 1), -1, dtype=torch.long, device=dev)
    j = torch.arange(L, device=dev).expand(B, L)
    src.scatter_(1, torch.where(active, dest, torch.full_like(dest, L)),
                 torch.where(active, j, torch.full_like(j, -1)))
    src = src[:, :L]

    keep = src >= 0
    gathered = x.gather(1, src.clamp_min(0)[..., None].expand(B, L, V))
    if noise is None:
        noise = torch.randn(B, L, V, device=dev, dtype=x.dtype).mul_(sigma)
    x_eps = torch.where(keep[..., None], gathered, noise)

    t_eps = torch.where(keep, t_local.gather(1, src.clamp_min(0)),
                        torch.zeros_like(t_local))
    n_tot = (active.sum(-1) + counts.sum(-1)).clamp(max=L)
    active_eps = torch.arange(L, device=dev)[None] < n_tot[:, None]
    return x_eps, active_eps, t_eps


@register("expand_training", "torch",
          note="Alg.2 L12 -- mask-only expansion, ZERO data movement")
def expand_training(state_x, t_ins, s, t):
    """Expand once to d(t): activity becomes ``t_ins <= t`` while local times are
    still taken at ``s``. Positions inserted over (s, t] are already sitting at
    their Gaussian latent with t_i = 0, so nothing moves. This is why the
    training path should never call expand_sampling."""
    active_t = t_ins <= t
    active_s = t_ins <= s
    t_local = torch.where(active_s,
                          ((s - t_ins) / (1 - t_ins).clamp_min(1e-7)).clamp(0, 1),
                          torch.zeros_like(t_ins))
    return state_x, active_t, t_local


# TODO(triton): fused permute + procedural noise fill. Traffic floor is
# 1R + 1W over ACTIVE rows only -- the -1 slots should never be written twice.
