r"""OP-5  Truncated insertion-count sampling (Eq. 28, Alg. 3 L4).

    l_i ~ Binomial(L - n_s,  Ihat_{s,t}[i] / (L - n_s))
    "truncate the joint draw left-to-right so that sum_i l_i <= L - n_s"

The paper states the truncation procedurally, and the obvious transcription is
a Python loop over the L+1 gaps that maintains a running budget.  At L=128 that
is 129 dependent kernel launches *per sampling step* -- for 1-step generation,
the headline use case, this single line would plausibly dominate wall clock.

It does not have to be sequential.  Left-to-right truncation is exactly

    S_i = min(S_{i-1} + l_i, B),   l'_i = S_i - S_{i-1},   S_{-1} = 0

and the family f_c(x) = min(x + c, B) with c >= 0 is *closed under
composition*:

    f_{c2}(f_{c1}(x)) = min(min(x + c1, B) + c2, B)
                      = min(x + c1 + c2,  B + c2,  B)
                      = min(x + (c1 + c2), B)          [since c2 >= 0]

so composing gap maps is just adding their counts.  Therefore

    S_i = min(cumsum(l)_i, B)

and the whole truncation collapses to a prefix sum, a clamp, and a diff -- no
scan of composed maps, no loop, three vectorised ops.  Verified exact against
the loop over 20k random cases; see tests/test_insertion.py.

This is the cheapest correctness-preserving speedup in the repo and it costs
nothing to take, so ``torch`` is already the fast form.  The remaining work is
fusing the three launches into one and generating the binomial draws inline.

Regime note (Eq. 28 discussion): many-step sampling has rho_{s,t} << 1 and
L - n_s >> Ihat, where the binomial converges to Poisson(Ihat) -- the PDMP jump
law of Sec. 3.  Few-step sampling drives rho -> 1 where the Poisson limit fails
(unbounded support, variance == mean) and the binomial concentrates
(variance np(1-p) -> 0).  Both are implemented; ``bench_ablations.py`` sweeps
them against step count because the correct choice is step-count dependent and
the cost profile differs.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register, requires_compile


@register("insertion_sample", "reference", reference=True)
def insertion_sample_reference(I_hat, budget, *, law="binomial", generator=None):
    """Literal transcription of Alg. 3 L4 including the left-to-right loop.

    I_hat  [B, G]  predicted per-gap expected insertion counts
    budget [B]     L - n_s, the remaining token budget
    """
    B, G = I_hat.shape
    out = torch.zeros(B, G, dtype=torch.long, device=I_hat.device)
    for b in range(B):
        n = int(budget[b].item())
        if n <= 0:
            continue
        used = 0
        for i in range(G):
            p = float(I_hat[b, i].clamp_min(0).item()) / max(n, 1)
            p = min(max(p, 0.0), 1.0)
            if law == "binomial":
                draw = int(torch.binomial(
                    torch.tensor([float(n)]), torch.tensor([p]),
                    generator=generator).item())
            else:
                draw = int(torch.poisson(I_hat[b, i].clamp_min(0).reshape(1),
                                         generator=generator).item())
            take = min(draw, n - used)
            out[b, i] = take
            used += take
    return out


@register("insertion_sample", "torch",
          note="loop -> cumsum+clamp+diff; exact, see module docstring")
def insertion_sample_torch(I_hat, budget, *, law="binomial", generator=None):
    B, G = I_hat.shape
    n = budget.clamp_min(0).to(I_hat.dtype)[:, None]              # [B,1]
    if law == "binomial":
        p = (I_hat.clamp_min(0) / n.clamp_min(1)).clamp(0.0, 1.0)
        draws = torch.binomial(n.expand(B, G).contiguous(), p, generator=generator)
    elif law == "poisson":
        draws = torch.poisson(I_hat.clamp_min(0), generator=generator)
    else:
        raise ValueError(law)

    cs = draws.cumsum(-1)
    capped = torch.minimum(cs, n)                                  # S_i
    prev = torch.nn.functional.pad(capped[:, :-1], (1, 0))         # S_{i-1}
    return (capped - prev).round().long()


@register("insertion_sample", "compile", available=requires_compile)
@torch.compile(dynamic=False)
def insertion_sample_compile(I_hat, budget, **kw):
    return insertion_sample_torch(I_hat, budget, **kw)
