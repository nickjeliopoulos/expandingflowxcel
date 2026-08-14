"""OP-2  Per-position expanding interpolant (Eq. 84/93, Alg. 1 L8, Alg. 2 L9).

    x_t^i = (1 - beta(t_i)) * z_i + beta(t_i) * onehot(x1_i),   z ~ N(0, sigma^2 I_V)

This is the first place the method becomes expensive, and it is expensive for a
reason that has nothing to do with arithmetic.

Traffic accounting at the LM1B config (B=128, L=128, V=30522, bf16):
    one [B,L,V] tensor          = 0.93 GiB
    naive PyTorch materialises  : one_hot (1 W), randn (1 W), 2 lerp temporaries
                                  -> ~4 writes + ~3 reads = ~6.5 GiB of traffic
    information-theoretic floor : 1 write = 0.93 GiB
                                  (labels are [B,L] int64 = 128 KiB; noise is
                                   generated, not read)

So there is a ~5-7x DRAM-traffic overhang here, on a purely memory-bound op.
A single Triton kernel that (a) draws z from a counter-based Philox stream
keyed by (seed, position) instead of reading it from DRAM, and (b) writes the
scaled noise everywhere while adding beta only at the label column, hits the
floor.  Expected 3-5x; this is the cheapest large win in the repo and should be
kernel #1.

Note the RNG contract: reproducibility across backends requires that the
Triton kernel and the reference agree on the noise *values*, not just the
distribution.  We therefore key the stream explicitly on (seed, b, l, v) and
test bitwise agreement in tests/test_backend_parity.py rather than comparing
moments.  Do not use torch.randn's global stream here.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register, requires_triton


def _beta(t_local: torch.Tensor, beta_fn=None) -> torch.Tensor:
    """Optional non-linear mixing coefficient of Eq. 85. Identity == linear."""
    return t_local if beta_fn is None else beta_fn(t_local)


@register("interpolant", "reference", reference=True)
def interpolant_reference(labels, t_local, active, V, sigma=1.0, *,
                          noise=None, beta_fn=None, dtype=torch.float64):
    """Ground truth. Materialises everything, float64, no cleverness.

    labels  [B, L] long      clean token ids  (x1, one-hot)
    t_local [B, L] float     per-position local time
    active  [B, L] bool
    noise   [B, L, V] or None -- pass explicitly to make the test deterministic
    """
    B, L = labels.shape
    if noise is None:
        noise = torch.randn(B, L, V, device=labels.device, dtype=dtype) * sigma
    noise = noise.to(dtype)
    x1 = torch.nn.functional.one_hot(labels, V).to(dtype)
    b = _beta(t_local.to(dtype), beta_fn)[..., None]
    x = (1 - b) * noise + b * x1
    # inactive positions sit at pure noise with t_i = 0 (App. C.1)
    return torch.where(active[..., None], x, noise)


@register("interpolant", "torch")
def interpolant_torch(labels, t_local, active, V, sigma=1.0, *,
                      noise=None, beta_fn=None, dtype=torch.bfloat16):
    B, L = labels.shape
    if noise is None:
        noise = torch.randn(B, L, V, device=labels.device, dtype=dtype) * sigma
    b = (_beta(t_local, beta_fn) * active).to(dtype)[..., None]
    x = noise.mul(1 - b)
    # scatter the beta mass onto the label column without building one_hot
    x.scatter_add_(-1, labels[..., None], b)
    return x


@register("interpolant", "compile")
@torch.compile(dynamic=False)
def interpolant_compile(labels, t_local, active, V, sigma=1.0, **kw):
    return interpolant_torch(labels, t_local, active, V, sigma, **kw)


@register("interpolant", "triton", available=requires_triton,
          note="kernel #1: fused philox-noise + scaled-scatter, target 3-5x")
def interpolant_triton(*a, **kw):
    raise NotImplementedError(
        "Contract: read labels[B,L] + t_local[B,L], write x[B,L,V] in ONE pass.\n"
        "  grid  = (B*L, cdiv(V, BLOCK_V))\n"
        "  noise = tl.randn(seed, offs) keyed on (b,l,v) -- never read from DRAM\n"
        "  write (1-beta)*sigma*noise, then a single scalar add of beta at v==label\n"
        "Validate with tests/test_backend_parity.py::test_interpolant_bitwise."
    )
