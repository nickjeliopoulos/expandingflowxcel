"""Insertion sampling: the scan reformulation must be EXACTLY the loop."""
from __future__ import annotations

import torch

from eflow.ops.insertion_sample import insertion_sample_torch


def loop_truncate(draws: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(draws)
    for b in range(draws.shape[0]):
        used, n = 0, int(budget[b])
        for i in range(draws.shape[1]):
            take = min(int(draws[b, i]), n - used)
            take = max(take, 0)
            out[b, i] = take
            used += take
    return out


def scan_truncate(draws: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    cs = draws.cumsum(-1)
    capped = torch.minimum(cs, budget[:, None])
    prev = torch.nn.functional.pad(capped[:, :-1], (1, 0))
    return capped - prev


def test_scan_equals_loop_exhaustively():
    """min(cumsum, B) differencing == left-to-right greedy truncation.
    Proof sketch in ops/insertion_sample.py; this is the empirical check."""
    g = torch.Generator().manual_seed(0)
    for _ in range(2000):
        B, G = 8, int(torch.randint(1, 16, (1,), generator=g))
        draws = torch.randint(0, 9, (B, G), generator=g)
        budget = torch.randint(0, 40, (B,), generator=g)
        assert torch.equal(loop_truncate(draws, budget), scan_truncate(draws, budget))


def test_budget_never_exceeded():
    g = torch.Generator().manual_seed(1)
    I = torch.rand(16, 33, generator=g) * 5
    budget = torch.randint(0, 30, (16,), generator=g)
    out = insertion_sample_torch(I, budget, generator=g)
    assert (out.sum(-1) <= budget).all()
    assert (out >= 0).all()


def test_binomial_mean_matches_head_prediction():
    """E[l_i] = Ihat_i when the budget is not binding (Eq. 28)."""
    g = torch.Generator().manual_seed(2)
    G = 8
    I = torch.full((20000, G), 1.5)
    budget = torch.full((20000,), 5000)           # far from binding
    out = insertion_sample_torch(I, budget, generator=g).double()
    assert torch.allclose(out.mean(0), torch.full((G,), 1.5, dtype=torch.float64), atol=0.05)


def test_binomial_concentrates_where_poisson_does_not():
    """Few-step regime (rho -> 1): binomial variance np(1-p) -> 0 while Poisson
    variance equals its mean. This is the stability claim of Eq. 28."""
    g = torch.Generator().manual_seed(3)
    n, G = 20000, 1
    budget = torch.full((n,), 10)
    I = torch.full((n, G), 10.0)                  # p -> 1
    binom = insertion_sample_torch(I, budget, law="binomial", generator=g).double()
    pois = insertion_sample_torch(I, budget, law="poisson", generator=g).double()
    assert binom.var() < pois.var()
