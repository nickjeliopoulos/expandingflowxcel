r"""OP-7  Semigroup consistency target + weighted cross-entropy.
       *** The flagship kernel target of this repository. ***

Equations 32, 33, 87 and Alg. 2 L15/L21:

    psibar_{s,t}(x)  = sg[ w * psihat^{(t)}_{s,u}(x) + (1-w) * psihat_{u,t}(Phi^{(t)}_{s,u}(x)) ]
    w_{s,u,t}        = (u-s)(1-t) / ((t-s)(1-u))
    L_DEFM           = - sum_i ( psibar_{s,t} . log psihat_{s,t} )(x^eps_s)_i
    weight           = sg[ (||psi_{s,t} - psibar_{s,t}||^2 + c)^{-r} ]   (Eq. 87, c=1e-6, r=0.5)

Why this dominates
------------------
At the LM1B config (B=128, L=128, V=30522, bf16) one [B,L,V] tensor is 0.93 GiB.
A naive autograd implementation of the above holds, simultaneously:

    psi_su, psi_ut, psibar, student logits, log_softmax, grad-wrt-logits
    = 6 x 0.93 GiB = 5.6 GiB

of live [B,L,V] activations for the *loss alone* -- comparable to the entire
rest of the model.  None of it is necessary.

The fused form
--------------
Backward through soft-target CE with a stop-gradient target is
    dL/dlogits = weight * (softmax(logits) - psibar)
which depends only on ``logits`` and ``psibar``.  So a kernel can:

  * stream over V in blocks, computing logsumexp online (no log_softmax tensor);
  * form ``psibar`` on the fly from psi_su / psi_ut as it reads them, instead of
    materialising the mixture;
  * accumulate ||psi - psibar||^2 for Eq. 87 in the *same* pass, so the adaptive
    weight costs no extra traffic (naively it is a second full reduction);
  * store nothing for backward and recompute from logits.

Residency drops 6x, to just the logits.  If the vocab projection is folded in
too (Cut-Cross-Entropy style: [B,L,D] @ [D,V] -> scalar, tiles of V never
leaving SRAM), logits are never materialised either and residency drops to the
[B,L,D] hidden states -- 0.023 GiB, a ~40x reduction.  That is the ceiling and
it is worth aiming straight at it.

Three-forward-pass structure
----------------------------
Alg. 2 needs psihat^{(t)}_{s,u}, psihat_{u,t}(Phi), psihat_{t,t} (diagonal
target) and the student psihat_{s,t}.  Two are under stop-grad.  So a training
step costs ~2 no-grad forwards + 1 forward/backward, i.e. roughly 3-4x a plain
denoiser step.  The two target forwards have identical shapes and differ only
in their (s,t) conditioning -- ``bench_step.py`` measures whether batching them
into a single 2B forward beats two B forwards (it should: better occupancy,
half the launches).  That is a free systems-level win requiring no kernel work.

Gradient surgery (E.3) merges L_DEFM and L_insert by projecting conflicting
gradients, which needs per-loss gradients and therefore either two backward
passes or a fused dual-accumulation.  Benchmarked separately in bench_step.py;
do not let it hide inside the loss timing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from eflow.ops.registry import register, requires_triton


def omega(s, t, u):
    """w_{s,u,t} of Prop. 5.1. Convex: 1 - w = (t-u)(1-s)/((t-s)(1-u))."""
    return ((u - s) * (1 - t)) / (((t - s) * (1 - u)).clamp_min(1e-7))


def flow_map_mix(x, psi, s, t):
    """Phi_{s,t}(x) = (1-t)/(1-s) x + (t-s)/(1-s) psi   (Eq. 30)."""
    d = (1 - s).clamp_min(1e-7)
    return ((1 - t) / d) * x + ((t - s) / d) * psi


@register("semigroup_ce", "reference", reference=True)
def semigroup_ce_reference(student_logits, psi_su, psi_ut, s, u, t, mask,
                           c=1e-6, r=0.5, adaptive_weight=True):
    """Ground truth: every intermediate materialised, float64, no fusion.

    student_logits [B,L,V]  pre-softmax output of psihat_{s,t}
    psi_su, psi_ut [B,L,V]  simplex-valued target components (already sg'd)
    mask           [B,L]    active positions only contribute (Eq. 101-102)
    """
    sl = student_logits.double()
    w = omega(s, t, u).double()
    while w.ndim < sl.ndim:
        w = w[..., None]

    psibar = (w * psi_su.double() + (1 - w) * psi_ut.double()).detach()
    logp = F.log_softmax(sl, dim=-1)
    psi = logp.exp()

    ce = -(psibar * logp).sum(-1)                       # [B, L]
    if adaptive_weight:
        delta2 = ((psi - psibar) ** 2).sum(-1)          # Eq. 87
        wt = (delta2 + c).pow(-r).detach()
        ce = ce * wt
    return (ce * mask).sum() / mask.sum().clamp_min(1)


@register("semigroup_ce", "torch")
def semigroup_ce_torch(student_logits, psi_su, psi_ut, s, u, t, mask,
                       c=1e-6, r=0.5, adaptive_weight=True):
    w = omega(s, t, u)
    while w.ndim < student_logits.ndim:
        w = w[..., None]
    psibar = torch.lerp(psi_ut, psi_su, w).detach()
    logp = F.log_softmax(student_logits, dim=-1)
    ce = -(psibar * logp).sum(-1)
    if adaptive_weight:
        delta2 = (logp.exp() - psibar).pow(2).sum(-1)
        ce = ce * (delta2 + c).pow(-r).detach()
    return (ce * mask).sum() / mask.sum().clamp_min(1)


@register("semigroup_ce", "chunked",
          note="chunk over L; bounds peak residency without a custom kernel")
def semigroup_ce_chunked(student_logits, psi_su, psi_ut, s, u, t, mask,
                         c=1e-6, r=0.5, adaptive_weight=True, chunk=16):
    """Baseline that any Triton kernel must beat. Same math, O(chunk) residency.
    Useful on its own: this alone should recover most of the memory win and
    tells us how much of the remaining gap isgenuinely kernel-level vs. scheduling."""
    B, L, _ = student_logits.shape
    total = student_logits.new_zeros((), dtype=torch.float32)
    for lo in range(0, L, chunk):
        hi = min(lo + chunk, L)
        total = total + semigroup_ce_torch(
            student_logits[:, lo:hi], psi_su[:, lo:hi], psi_ut[:, lo:hi],
            s, u, t, mask[:, lo:hi], c, r, adaptive_weight
        ) * mask[:, lo:hi].sum()
    return total / mask.sum().clamp_min(1)


@register("semigroup_ce", "triton", available=requires_triton,
          note="FLAGSHIP: online-softmax soft-target CE, fused mixture + Eq.87 "
               "reduction, recompute-backward. Target 6x memory, 2-3x time.")
def semigroup_ce_triton(*a, **kw):
    raise NotImplementedError(
        "Contract:\n"
        "  fwd: one pass over V per (b,l) tile. Online logsumexp; read psi_su/psi_ut\n"
        "       once, form psibar in registers; accumulate BOTH -sum(psibar*logp)\n"
        "       and sum((psi-psibar)^2) in the same pass. Save NOTHING but logits.\n"
        "  bwd: dL/dlogits = weight * (softmax(logits) - psibar), recomputed.\n"
        "  stretch: fold the [D,V] vocab projection in (Cut-Cross-Entropy) so\n"
        "           logits never hit DRAM -> residency becomes [B,L,D].\n"
        "Parity target: semigroup_ce_reference in float64, rtol 1e-3 on the loss\n"
        "and 1e-3 on d(loss)/d(logits)."
    )
