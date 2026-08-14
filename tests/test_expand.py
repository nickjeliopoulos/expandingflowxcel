"""Expand operator: the vectorised path must equal the literal Alg. 3 loop.

This is the fiddliest function in the repo -- it inverts a gap-offset map to
build a scatter index -- so it gets the most direct test: same inputs, same
pinned noise, compare all three outputs against the loop transcription.

Noise convention: a caller-supplied `noise` tensor is treated as ALREADY scaled
by sigma. Both backends must honour that or the comparison is meaningless.
"""
from __future__ import annotations

import pytest
import torch

from eflow.ops.expand import (exclusive_cumsum, expand_sampling_reference,
                              expand_sampling_torch, expand_training)


def _case(B=3, L=12, V=5, seed=0, device="cpu"):
    """Random state + counts that respect the budget sum_i l_i <= L - n_s."""
    g = torch.Generator(device=device).manual_seed(seed)
    active = torch.rand(B, L, generator=g, device=device) > 0.5
    x = torch.randn(B, L, V, generator=g, device=device, dtype=torch.float64)
    t_local = torch.rand(B, L, generator=g, device=device, dtype=torch.float64) * active
    noise = torch.randn(B, L, V, generator=g, device=device, dtype=torch.float64)

    counts = torch.zeros(B, L + 1, dtype=torch.long, device=device)
    for b in range(B):
        budget = L - int(active[b].sum())
        n_gaps = int(active[b].sum()) + 1
        left = budget
        for i in range(n_gaps):
            if left <= 0:
                break
            take = int(torch.randint(0, left + 1, (1,), generator=g, device=device))
            counts[b, i] = take
            left -= take
    return x, t_local, active, counts, noise


@pytest.mark.parametrize("seed", range(8))
def test_vectorised_expand_matches_loop(seed):
    x, t_local, active, counts, noise = _case(seed=seed)
    ref_x, ref_a, ref_t = expand_sampling_reference(
        x, t_local, active, counts, noise=noise)
    got_x, got_a, got_t = expand_sampling_torch(
        x, t_local, active, counts, noise=noise)

    torch.testing.assert_close(got_x.double(), ref_x.double(), rtol=0, atol=1e-12)
    assert torch.equal(got_a, ref_a), "activity mask diverged"
    torch.testing.assert_close(got_t.double(), ref_t.double(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_expand_preserves_order_and_content(seed):
    """Every active row must survive exactly once, in its original order
    (Eq. 27 concatenates gaps and old tokens left to right)."""
    x, t_local, active, counts, noise = _case(seed=seed)
    got_x, got_a, _ = expand_sampling_torch(x, t_local, active, counts, noise=noise)
    for b in range(x.shape[0]):
        old = x[b][active[b]]
        # rows of the output that are not noise, in order
        kept = [r for r in got_x[b] if not any(torch.equal(r, n) for n in noise[b])]
        assert len(kept) == len(old), f"expected {len(old)} survivors, got {len(kept)}"
        for a_, b_ in zip(kept, old):
            torch.testing.assert_close(a_, b_, rtol=0, atol=1e-12)


def test_new_tokens_start_at_local_time_zero():
    x, t_local, active, counts, noise = _case(seed=3)
    _, got_a, got_t = expand_sampling_torch(x, t_local, active, counts, noise=noise)
    n_old = int(active[0].sum())
    assert (got_t[0][got_a[0]] >= 0).all()
    assert int((got_t[0] > 0).sum()) <= n_old, "a freshly inserted token has t_i > 0"


def test_budget_overflow_does_not_corrupt():
    """Counts exceeding the budget are a caller error; the clamp must still
    produce a valid (if truncated) state rather than aliasing rows together."""
    x, t_local, active, counts, noise = _case(seed=5)
    counts = counts + 50                      # deliberately absurd
    got_x, got_a, got_t = expand_sampling_torch(x, t_local, active, counts, noise=noise)
    assert got_x.shape == x.shape and torch.isfinite(got_x).all()
    assert got_a.all(), "buffer should be saturated when counts exceed the budget"


def test_exclusive_cumsum():
    c = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(exclusive_cumsum(c), torch.tensor([[0, 1, 3, 6]]))


def test_training_expand_moves_no_data():
    """Alg. 2 L12: expanding to d(t) during training is a mask change only."""
    B, L, V = 2, 10, 4
    x = torch.randn(B, L, V)
    t_ins = torch.rand(B, L)
    s, t = torch.tensor(0.3), torch.tensor(0.7)
    out_x, active_t, t_local = expand_training(x, t_ins, s, t)
    assert out_x is x, "training expand must not copy the state"
    assert torch.equal(active_t, t_ins <= t)
    assert torch.equal(t_local[t_ins > s], torch.zeros_like(t_local[t_ins > s]))
