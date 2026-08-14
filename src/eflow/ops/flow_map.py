r"""OP-6  Mean denoiser -> flow map affine combination (Eq. 29, 30, 100).

    psi_{s,t}(x) = softmax(logits)                       (simplex, Eq. 29)
    Phi_{s,t}(x) = (1-t)/(1-s) * x  +  (t-s)/(1-s) * psi (Eq. 30)

Memory-bound over [B,L,V].  Naive PyTorch: softmax (2 passes over V) + 2 muls +
1 add ~= 5 passes.  Fused: 1 pass, reading logits and x, writing Phi.

Backward through the softmax is a rank-1 correction
    dL/dlogits = psi * (g - (psi . g))     with  g = (t-s)/(1-s) * dL/dPhi
which fuses into the same streaming pattern, so the whole op should be 1 pass
forward and 1 pass backward with no saved [B,L,V] tensor beyond logits.

The continuous variant (Eq. 18) has no softmax:
    Phi_{s,t}(x) = E_{s,t}(x) + (t-s) * v_{s,t}(E_{s,t}(x))
and is a plain fused-multiply-add -- registered separately as ``flow_map_cont``
so the two do not get benchmarked as one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from eflow.ops.registry import register, requires_compile, requires_triton


@register("flow_map", "reference", reference=True)
def flow_map_reference(logits, x, s, t):
    psi = F.softmax(logits.double(), dim=-1)
    d = (1 - s).clamp_min(1e-7)
    return ((1 - t) / d) * x.double() + ((t - s) / d) * psi, psi


@register("flow_map", "torch")
def flow_map_torch(logits, x, s, t):
    psi = F.softmax(logits, dim=-1)
    d = (1 - s).clamp_min(1e-7)
    return ((1 - t) / d) * x + ((t - s) / d) * psi, psi


@register("flow_map", "compile", available=requires_compile)
@torch.compile(dynamic=False)
def flow_map_compile(logits, x, s, t):
    return flow_map_torch(logits, x, s, t)


@register("flow_map_cont", "torch")
def flow_map_cont(x_expanded, v, s, t):
    """Continuous EFM, Eq. 18. Pure FMA over [B, A, 3]."""
    return x_expanded + (t - s) * v


@register("flow_map", "triton", available=requires_triton,
          note="fused softmax + affine, 5 passes -> 1")
def flow_map_triton(*a, **kw):
    raise NotImplementedError("see module docstring for the streaming contract")
