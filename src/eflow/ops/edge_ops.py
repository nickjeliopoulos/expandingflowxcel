r"""OP-9  Graph edge operators (App. D.2, D.3, D.6).

    a_ij = a_i * a_j * (1 - delta_ij)          (Eq. 92)
    t_ij = min(t_i, t_j)                       (Eq. 92)
    E    <- (E + E^T)/2,  diag(E) = 0          (Eq. 94, D.6)

The edge tensor is [B, L, L, Ve].  At QM9 scale (L=9, Ve=5) this is noise; the
reason it is a first-class op is the *scaling* question -- GEOM-Drugs reaches
181 atoms, and the paper's own limitations section flags scaling as the open
problem.  So sweep L in {9, 32, 64, 128, 181} and watch the quadratic term
overtake everything.

Symmetrisation is the interesting one: (E + E^T)/2 on the *first two* axes of a
4-D tensor is a strided transpose-add with a pathological access pattern, and it
is applied after *every* jump (D.6).  A tiled kernel that loads an (i,j) block
and its (j,i) partner together turns two uncoalesced streams into one.  Also
worth testing: store only the upper triangle and never symmetrise -- that halves
both traffic and memory and makes the invariant structural rather than enforced,
at the cost of indexing complexity in the denoiser. That is a real design
question, not just a kernel one.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register


@register("edge_time", "reference", reference=True)
def edge_time_reference(t_local, active):
    B, L = t_local.shape
    t = torch.zeros(B, L, L, dtype=torch.float64, device=t_local.device)
    a = torch.zeros(B, L, L, dtype=torch.bool, device=t_local.device)
    for b in range(B):
        for i in range(L):
            for j in range(L):
                if i != j and active[b, i] and active[b, j]:
                    a[b, i, j] = True
                    t[b, i, j] = min(float(t_local[b, i]), float(t_local[b, j]))
    return t, a


@register("edge_time", "torch")
def edge_time_torch(t_local, active):
    t = torch.minimum(t_local[:, :, None], t_local[:, None, :])
    a = active[:, :, None] & active[:, None, :]
    eye = torch.eye(active.shape[1], dtype=torch.bool, device=active.device)
    a = a & ~eye
    return t * a, a


@register("symmetrize", "torch")
def symmetrize_torch(E):
    """E <- (E + E^T)/2 on axes (1,2); zero the diagonal. Invariant of D.6."""
    E = 0.5 * (E + E.transpose(1, 2))
    L = E.shape[1]
    eye = torch.eye(L, dtype=torch.bool, device=E.device)
    return E.masked_fill(eye[None, :, :, None], 0.0)


@register("symmetrize", "reference", reference=True)
def symmetrize_reference(E):
    out = E.double().clone()
    L = E.shape[1]
    for i in range(L):
        for j in range(L):
            out[:, i, j] = 0.0 if i == j else 0.5 * (E[:, i, j] + E[:, j, i])
    return out
