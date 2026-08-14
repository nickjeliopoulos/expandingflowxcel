"""Canonical state containers for Expanding Flow Maps.

Design decision that everything else depends on
-----------------------------------------------
The paper writes the state as a *ragged* object: ``x_s`` has length ``d(s)``
and ``x_s^eps`` has length ``d(t)`` (Eq. 24, Alg. 2 L11-12).  We do **not**
represent it that way.  Instead every EFMState carries a fixed-size buffer of
length ``L`` (the maximum length / node budget ``d_max``) plus an activity mask.

Why:
  * dynamic shapes defeat CUDA graphs, torch.compile, and any stable
    benchmark measurement -- every step would recompile;
  * Alg. 2 L12 already "expands once to d(t)" during training, so the ragged
    form buys nothing on the training path;
  * the ragged form is recovered exactly by ``active``, so nothing is lost.

The cost of this choice is padding waste, which is itself one of the headline
things we want to measure (see ``bench/bench_ablations.py::padded_vs_varlen``).
Do not "fix" it silently -- it is an experimental variable.

Shape conventions
-----------------
B  batch                       L  buffer length (== L_max or d_max)
V  vocabulary / category count D  model hidden size
All float tensors default to ``torch.bfloat16`` on the benchmark path and
``torch.float64`` on the reference path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass
class SeqState:
    """Discrete-sequence EFM state (App. C).

    Attributes
    ----------
    x        [B, L, V]  simplex-valued / Gaussian-latent token states.
    active   [B, L]     bool. ``True`` iff ``t_ins_i <= t`` (Eq. 90).
    t_local  [B, L]     per-position local time ``t_i`` of Eq. 12/83.
                        Zero (not undefined) at inactive positions.
    t_ins    [B, L]     per-position insertion time. ``> 1`` marks a slot that
                        never activates (padding), following App. D.2.
    t_global scalar or [B] -- the global time this state is materialised at.
    """

    x: torch.Tensor
    active: torch.Tensor
    t_local: torch.Tensor
    t_ins: torch.Tensor
    t_global: torch.Tensor

    @property
    def n_active(self) -> torch.Tensor:  # [B] -- this is d(t)
        return self.active.sum(-1)

    def replace(self, **kw) -> "SeqState":
        return replace(self, **kw)

    def assert_consistent(self) -> None:
        B, L, _ = self.x.shape
        assert self.active.shape == (B, L)
        assert self.t_local.shape == (B, L)
        # local time must vanish exactly where the position is inactive
        assert torch.equal(self.t_local[~self.active],
                           torch.zeros_like(self.t_local[~self.active]))
        # activity must agree with the insertion times it was built from
        tg = self.t_global.reshape(-1, 1) if self.t_global.ndim else self.t_global
        assert torch.equal(self.active, self.t_ins <= tg)


@dataclass
class GraphState:
    """Discrete-graph EFM state (App. D).

    ``E`` is kept symmetric with zero diagonal at *all* times; every op that
    touches it must restore that invariant before returning (App. D.6).
    Edge local time is *derived*, never stored: ``t_ij = min(t_i, t_j)``
    (Eq. 92).  Storing it would be a [B,L,L] tensor we would have to keep in
    sync -- deriving it is cheap and cannot go stale.
    """

    x: torch.Tensor          # [B, L, Vx]
    E: torch.Tensor          # [B, L, L, Ve]
    active: torch.Tensor     # [B, L]
    t_local: torch.Tensor    # [B, L]
    t_ins: torch.Tensor      # [B, L]
    t_global: torch.Tensor

    def edge_active(self) -> torch.Tensor:  # [B, L, L], Eq. 92
        a = self.active
        m = a[:, :, None] & a[:, None, :]
        return m & ~torch.eye(a.shape[1], dtype=torch.bool, device=a.device)

    def edge_time(self) -> torch.Tensor:    # [B, L, L], Eq. 92
        t = self.t_local
        return torch.minimum(t[:, :, None], t[:, None, :])


@dataclass
class PointCloudState:
    """Continuous coarse-to-fine EFM state (App. E.1).

    The conformer path differs from the discrete ones in that the expand
    operator is *deterministic* (child expansion, Sec. 3(iii)) -- the atom
    count is fixed by the molecular graph, so there is no learned insertion
    head and no count to infer at sampling time.  ``parent`` encodes the
    child->parent map used by Eq. (iii): x0_i = x0_{pa(i)} + sigma_h * eps.
    """

    x: torch.Tensor            # [B, A, 3]
    active: torch.Tensor       # [B, A]
    t_local: torch.Tensor      # [B, A]
    t_ins: torch.Tensor        # [B, A]
    parent: torch.Tensor       # [B, A] long; self-index for heavy atoms
    batch_ptr: Optional[torch.Tensor] = None   # ragged CSR offsets, if used
    t_global: Optional[torch.Tensor] = None
