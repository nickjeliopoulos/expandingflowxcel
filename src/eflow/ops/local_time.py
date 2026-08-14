"""OP-1  Local time coordinate (Eq. 12 / 83, Alg. 1 L7, Alg. 2 L8).

    t_i = max(0, (t - t_ins_i) / (1 - t_ins_i)),   a_i = 1{t_ins_i <= t}

Elementwise over [B, L]; ~0 FLOPs and ~0 bytes on its own.  It is registered as
an op anyway because it is the *dependency* that every fused kernel downstream
has to absorb: if the interpolant kernel recomputes t_i inline from t_ins it
saves a full [B,L] round trip and, more importantly, removes a kernel launch
from the critical path of a 1-step sampling call where launch latency dominates.

Benchmark it standalone only to establish the launch-overhead floor.
"""
from __future__ import annotations

import torch

from eflow.ops.registry import register, requires_compile, requires_triton

_EPS = 1e-7


@register("local_time", "reference", reference=True)
def local_time_reference(t_ins: torch.Tensor, t: torch.Tensor):
    """Loop-free but float64 and deliberately unfused. Ground truth."""
    t_ins = t_ins.double()
    t = torch.as_tensor(t, dtype=torch.float64, device=t_ins.device)
    if t.ndim == 1:
        t = t[:, None]
    active = t_ins <= t
    t_local = (t - t_ins) / (1.0 - t_ins).clamp_min(_EPS)
    t_local = torch.where(active, t_local.clamp(0.0, 1.0), torch.zeros_like(t_local))
    return t_local, active


@register("local_time", "torch")
def local_time_torch(t_ins: torch.Tensor, t: torch.Tensor):
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=t_ins.dtype, device=t_ins.device)
    if t.ndim == 1:
        t = t[:, None]
    active = t_ins <= t
    t_local = ((t - t_ins) / (1.0 - t_ins).clamp_min(_EPS)).clamp(0.0, 1.0)
    return t_local * active, active


@register("local_time", "compile", available=requires_compile)
@torch.compile(dynamic=False)
def local_time_compile(t_ins, t):
    return local_time_torch(t_ins, t)


# TODO(triton): fold this into interpolant_triton rather than shipping it alone.
