"""Every registered backend must match its op's reference backend.

This file is intentionally generic: adding a Triton kernel to ops/ gives it a
correctness test automatically, with no edit here. That is the whole point of
the registry.

Tolerances are per-op because a bf16 streaming kernel cannot match a float64
reference to 1e-12 and pretending otherwise leads to tests being disabled.
"""
from __future__ import annotations

import pytest
import torch

from eflow.ops import registry

registry.load_all()

TOL = {                       # (rtol, atol) against the float64 reference
    "local_time":       (1e-6, 1e-6),
    "interpolant":      (2e-2, 2e-2),   # bf16
    "gap_counts":       (0, 0),         # integer: exact
    "insertion_sample": (0, 0),         # exact given a fixed generator
    "flow_map":         (2e-2, 2e-2),
    "semigroup_ce":     (1e-3, 1e-3),
    "adaln":            (2e-2, 2e-2),
    "edge_time":        (1e-6, 1e-6),
    "symmetrize":       (1e-6, 1e-6),
    "child_expand":     (1e-6, 1e-6),
    "com_free":         (1e-6, 1e-6),
    "poisson_nll":      (1e-6, 1e-6),
}


def _inputs(op, device):
    """Small, deterministic inputs per op. Kept tiny so the O(B*L) python
    reference implementations stay tractable."""
    g = torch.Generator(device=device).manual_seed(0)
    B, L, V, D = 2, 6, 9, 16
    if op == "local_time":
        return (torch.rand(B, L, generator=g, device=device),
                torch.tensor([0.5], device=device)), {}
    if op == "gap_counts":
        return (torch.rand(B, L, generator=g, device=device) > 0.4,), {}
    if op == "symmetrize":
        return (torch.randn(B, L, L, 3, generator=g, device=device),), {}
    if op == "edge_time":
        return (torch.rand(B, L, generator=g, device=device),
                torch.rand(B, L, generator=g, device=device) > 0.3), {}
    if op == "com_free":
        return (torch.randn(B, L, 3, generator=g, device=device),
                torch.rand(B, L, generator=g, device=device) > 0.3), {}
    if op == "poisson_nll":
        return (torch.randint(0, 4, (B, L), generator=g, device=device).double(),
                torch.rand(B, L, generator=g, device=device).double() + 0.1), {}
    pytest.skip(f"no fixture for {op} yet -- add one before landing a kernel")


@pytest.mark.parametrize("op", sorted(TOL))
def test_backends_match_reference(op):
    if op not in list(registry.ops()):
        pytest.skip(f"{op} not registered")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, kw = _inputs(op, device)
    ref = registry.reference_of(op)
    expected = ref.fn(*args, **kw)
    rtol, atol = TOL[op]
    for impl in registry.backends(op):
        if impl.reference:
            continue
        why = impl.available()
        if why:
            pytest.skip(why)
        got = impl.fn(*args, **kw)
        e = expected if torch.is_tensor(expected) else expected[0]
        gtens = got if torch.is_tensor(got) else got[0]
        torch.testing.assert_close(gtens.double(), e.double(), rtol=rtol, atol=atol,
                                   msg=f"{op}/{impl.backend} disagrees with reference")
