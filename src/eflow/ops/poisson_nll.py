"""OP-4b  Poisson NLL for the insertion head (Eq. 26, Alg. 1 L13, Alg. 2 L20).

    phi(a, b) = b - a + a * log(a / b)

argmin_b E[phi(A, b)] = E[A], which is the whole point: the head is trained to
regress a conditional *mean count* by being supervised against a realised count.

Numerically this is the fragile op in the loss stack.  a = 0 is common (most
gaps are empty most of the time) and a*log(a/b) -> 0 there, but the naive
expression evaluates log(0) first.  b -> 0 also blows up.  The reference clamps
explicitly; any kernel must reproduce the same clamping or parity tests will
fail in exactly the tail cases that matter.

Shape [B, L+1] -- negligible cost.  Registered for correctness coverage and
because it must be fused into the insertion-head epilogue eventually.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register

_EPS = 1e-8


@register("poisson_nll", "reference", reference=True)
def poisson_nll_reference(a, b, eps: float = _EPS):
    a = a.double(); b = b.double().clamp_min(eps)
    term = torch.where(a > 0, a * (torch.log(a.clamp_min(eps)) - torch.log(b)),
                       torch.zeros_like(a))
    return b - a + term


@register("poisson_nll", "torch")
def poisson_nll_torch(a, b, eps: float = _EPS):
    b = b.clamp_min(eps)
    term = torch.where(a > 0, a * (torch.log(a.clamp_min(eps)) - torch.log(b)),
                       torch.zeros_like(a))
    return b - a + term
