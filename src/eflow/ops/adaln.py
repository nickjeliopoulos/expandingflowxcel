r"""OP-8  Per-position adaLN modulation (App. C.3).

Standard DiT conditions each block on a single vector c in R^D and broadcasts
six modulation params over the sequence.  EFM cannot: every position carries its
*own* local time t_i, so the source-time embedding is per-position while the
target-time embedding stays global (App. C.3).  Modulation becomes

    (shift1, scale1, gate1, shift2, scale2, gate2) : [B, L, 6D]   not  [B, 6D]

This is the most under-advertised cost in the paper and one of the clearest
architecture/systems co-design levers, which is why it gets its own op.

Measured cost at the LM1B config (B=128, L=128, D=768, cond=128, 12 blocks):

    per-block core (attn qkvo + mlp + scores) : 238.4 GFLOP
    modulation, broadcast (standard DiT)      :   0.15 GFLOP   (0.06%)
    modulation, per-position (EFM)            :  19.3  GFLOP   (8.1%)

    modulation activations, broadcast         :  13.5 MiB total
    modulation activations, per-position      :   1.69 GiB total   (~128x)

So ~8% extra FLOPs but ~1.7 GiB of extra live activations, and it converts a
free broadcast into a materialised elementwise op on the largest hidden tensor
in the block.  Levers worth benchmarking (bench_ablations.py):

  (a) fused LN + per-position scale/shift and fused gate + residual add
      (the broadcast variants exist in Liger/Apex; the per-position one does not)
  (b) low-rank modulation: Linear(cond -> k -> 6D) with k << cond
  (c) share modulation across blocks, or across position groups with equal t_i
  (d) exploit that t_i takes few *distinct* values under a fixed insertion grid
      -- modulation could be computed once per distinct time and gathered

(d) is the interesting one: under a fixed schedule the number of distinct local
times at a given global t is bounded by the number of insertion events so far,
which is << L early in the trajectory. That turns a [B,L,6D] GEMM into a small
GEMM plus a gather.

Note the zero-init contract from App. C.3: the *target*-time embedder's final
projection is zero-initialised so the model starts as a single-time denoiser
conditioned on s. Any kernel must preserve that at init or early distillation
destabilises.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eflow.ops.registry import register, requires_compile, requires_triton


def modulate(x, shift, scale):
    """x * (1 + scale) + shift, with shift/scale either [B,1,D] or [B,L,D]."""
    return x * (1 + scale) + shift


@register("adaln", "reference", reference=True)
def adaln_reference(x, shift, scale, weight=None, bias=None, eps=1e-6):
    x = x.double()
    h = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return modulate(h, shift.double(), scale.double())


@register("adaln", "torch")
def adaln_torch(x, shift, scale, weight=None, bias=None, eps=1e-6):
    h = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return modulate(h, shift, scale)


@register("adaln", "compile", available=requires_compile)
@torch.compile(dynamic=False)
def adaln_compile(x, shift, scale, weight=None, bias=None, eps=1e-6):
    return adaln_torch(x, shift, scale, weight, bias, eps)


@register("adaln", "triton", available=requires_triton,
          note="fused LN + per-position affine; bwd must reduce over B,L for "
               "d(scale)/d(shift) -- that reduction is the hard part")
def adaln_triton(*a, **kw):
    raise NotImplementedError(
        "Contract: one pass, LN stats + affine in registers.\n"
        "Backward is where the per-position variant diverges from the broadcast\n"
        "one: dscale/dshift are [B,L,D] (no reduction) rather than [B,D], which\n"
        "REMOVES the cross-position reduction that dominates the standard kernel.\n"
        "Expect the per-position backward to be *easier*, not harder."
    )


class DualTimeEmbedder(nn.Module):
    """Two sinusoidal embedders per App. C.3: per-position source time, global
    target time, summed. Target projection zero-init."""

    def __init__(self, cond_dim=128, hidden=768, n_mod=6):
        super().__init__()
        self.cond_dim = cond_dim
        self.src = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.tgt = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.proj = nn.Linear(cond_dim, n_mod * hidden)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.tgt_proj = nn.Linear(cond_dim, cond_dim)
        nn.init.zeros_(self.tgt_proj.weight); nn.init.zeros_(self.tgt_proj.bias)

    def sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.cond_dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) *
                          (torch.log(torch.tensor(10000.0)) / half))
        a = t[..., None].float() * freqs
        return torch.cat([a.cos(), a.sin()], dim=-1)

    def forward(self, t_local, t_target):
        """t_local [B,L] -> per-position; t_target [B] -> broadcast."""
        c = self.src(self.sinusoidal(t_local))                     # [B,L,C]
        c = c + self.tgt_proj(self.tgt(self.sinusoidal(t_target)))[:, None]
        return self.proj(c)                                        # [B,L,6D]
